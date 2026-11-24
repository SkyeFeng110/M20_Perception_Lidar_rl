"""Prepare an isolated copy of the actual 1516 receiver-reset native release.

Never starts control. The three modified controller/fixture sources must be
rebuilt; unchanged dependencies may be reused with their recorded hashes.
"""
import argparse
import json
from pathlib import Path
import shutil
from release_utils import digest, replace_once, refresh_checksums

SOURCE_HASHES = {
 'run_policy/m20_policy_runner.hpp': 'df46d341fc714f426ac0bd5c4d441075f49052c214534b4a3ffa83bdac30bdc6',
 'state_machine/quadruped_wheel/standup_state.hpp': '0c5df2b05e8d07f9adfe12f9af2b1c5b665efc9890530110d529b997bf297597',
 'scripts/policy_smoke_test.cpp': '718ebd2dbfe822c08c8398fa98851c3c0ca8692d4ae733b9a47bb7309f628bed',
 'scripts/policy_alignment_check.cpp': '0d68b29c301df5a937cfbe1eb7ac0483a5876f01ec7ff316a787c3931e8cddc1'}
BASELINE_ONNX = '1762caa40a2ae0d4cd1cc63764b9fe0fd61134c57439e790dac0255a93befc32'


def patch_alignment_checker(text):
    text=replace_once(text,'#include "v11_observation_store.hpp"',
        '#include "v11_observation_store.hpp"\n#include "trained_execution_contract.hpp"')
    return replace_once(text, 'ok &= close(post,wheel_stop&&k>=12?0.:raw);',
        'const double clipped=std::max(-double(m20::trained::action_clip),std::min(double(m20::trained::action_clip),raw));ok &= close(post,wheel_stop&&k>=12?0.:clipped);')


def prepare(release, export):
    src = release/'src/M20_sdk_deploy'
    report = json.loads((export/'model.onnx.provenance.json').read_text())
    for path, expected in [(export/'model.onnx', report['deployment_sha256']),
                           (export/'trained_execution_contract.hpp', report['execution_header_sha256']),
                           (release/'policy/model.onnx', BASELINE_ONNX)]:
        if digest(path) != expected:
            raise ValueError('Artifact hash mismatch: '+str(path))
    if report['baseline_onnx_sha256'] != BASELINE_ONNX:
        raise ValueError('Not the actual 1516 reference model')
    for name, expected in SOURCE_HASHES.items():
        if digest(src/name) != expected:
            raise ValueError('Unexpected native revision: '+name)
    for name, expected in report['config_sha256'].items():
        if digest(export/'training_params'/name) != expected:
            raise ValueError('Training config hash mismatch: '+name)
    manifest = json.loads((release/'AOS_RELEASE_MANIFEST.json').read_text())
    if not manifest.get('explicit_receiver_reset') or manifest.get('observation_source') != 'staircase':
        raise ValueError('Receiver-reset staircase release required')
    if (release/'build').exists():
        raise ValueError('Stage a new copy without build/')
    p = src/'run_policy/m20_policy_runner.hpp'
    text = replace_once(p.read_text(), '#include "policy_alignment_trace.hpp"',
                        '#include "policy_alignment_trace.hpp"\n#include "trained_model_contract.hpp"')
    start = text.index('        dof_default_eigen_policy <<'); end = text.index('        SetDecimation(4);', start)
    text = text[:start]+'''        for (int i=0; i<action_dim; ++i) {
            dof_default_eigen_policy(i)=m20::trained::default_policy[i];
            dof_default_eigen_robot(i)=m20::trained::default_robot[i];
        }
'''+text[end:]
    anchor = '''        if (observation_mode_ == ObservationMode::kV11TwoRiser) {
            static constexpr const char* kLegacyContract'''
    text = replace_once(text, anchor, '''        if (observation_dim_ != static_cast<int>(m20::trained::obs_low.size()) ||
            observation_mode_ != ObservationMode::kV11TwoRiser)
            throw std::runtime_error("This release requires the 63D V11 ABI");
        {
            auto meta=session_.GetModelMetadata();
            m20::ValidateTrainedModel([&](const char* key) {
                auto value=meta.LookupCustomMetadataMapAllocated(key,allocator);
                return value == nullptr ? std::string{} : std::string(value.get());
            });
        }
'''+anchor)
    text = replace_once(text, '        alignment_observation_built_=current_observation_.size()==63;', '''        // The original finite rejection remains before clipping.
        for (int i=0; i<observation_dim_; ++i)
            current_observation_(i)=std::max(m20::trained::obs_low[i],
                std::min(m20::trained::obs_high[i],current_observation_(i)));
        alignment_observation_built_=current_observation_.size()==63;''')
    text = replace_once(text, '        warned_invalid_action_ = false;\n\n        if (zero_command_wheel_stop_', '''        warned_invalid_action_ = false;
        // Preserve raw target overflow rejection before action clipping.
        for (int i=0; i<action_dim; ++i)
            if (!std::isfinite(current_action_eigen(policy2robot_idx[i])*action_scale_robot[i]
                               +dof_default_eigen_robot(i))) return InterimV11DampingAction();
        current_action_eigen=current_action_eigen.cwiseMax(-m20::trained::action_clip)
                                .cwiseMin(m20::trained::action_clip);

        if (zero_command_wheel_stop_''')
    p.write_text(text)
    p=src/'state_machine/quadruped_wheel/standup_state.hpp'
    text=replace_once(p.read_text(),'#include "state_base.h"','#include "state_base.h"\n#include "trained_execution_contract.hpp"')
    start=text.index('            policy_default_joint_pos_ = Vec4f(');end=text.index('            Vec4f one_leg_kp',start)
    p.write_text(text[:start]+'''            policy_default_joint_pos_.resize(16);
            for (int i=0; i<16; ++i) policy_default_joint_pos_(i)=m20::trained::default_robot[i];

'''+text[end:])
    p=src/'scripts/policy_smoke_test.cpp';text=p.read_text()
    start=text.index('        default_position <<');end=text.index('        for (int step',start)
    p.write_text(text[:start]+'''        for (int i=0; i<16; ++i) default_position(i)=m20::trained::default_robot[i];
'''+text[end:])
    p=src/'scripts/policy_alignment_check.cpp';p.write_text(patch_alignment_checker(p.read_text()))
    for name in ('model.onnx','model.onnx.sha256','model.onnx.feature_contract','model.onnx.provenance.json'):
        shutil.copy2(export/name,release/'policy'/name)
    shutil.copytree(export/'training_params',release/'policy/training_params')
    shutil.copy2(export/'trained_execution_contract.hpp',src/'run_policy/trained_execution_contract.hpp')
    meta=json.loads((export/'trained_metadata.json').read_text())
    header=['#pragma once','#include "trained_execution_contract.hpp"','#include <string>',
            '#include <sstream>','#include <cmath>','#include <stdexcept>',
            'namespace m20 { template<class Getter> void ValidateTrainedModel(Getter get) {']
    for key in ('base_link_name','joint_names','m20_feature_contract'):
        header.append('if(get(%s)!=%s) throw std::runtime_error("Model metadata differs: %s");'%
                      (json.dumps(key),json.dumps(meta[key]),key))
    for key in ('default_joint_pos','joint_stiffness','joint_damping','action_scale'):
        values=[float(v) for v in meta[key].split(',')]
        expected='trained::default_policy' if key=='default_joint_pos' else 'std::array<float,16>{'+','.join('%.9ff'%v for v in values)+'}'
        header.append('''{const auto expected=%s; std::istringstream stream(get("%s")); std::string item; size_t i=0;
while(std::getline(stream,item,',')){size_t used=0;float v=std::stof(item,&used);
if(used!=item.size()||i>=expected.size()||!std::isfinite(v)||std::abs(v-expected[i])>1e-5f)
throw std::runtime_error("Model numeric metadata differs: %s");++i;}
if(i!=expected.size())throw std::runtime_error("Missing numeric metadata: %s");}'''%(expected,key,key,key))
    header.append('}}');(src/'run_policy/trained_model_contract.hpp').write_text('\n'.join(header)+'\n')
    shutil.copy2(release/'AOS_RELEASE_MANIFEST.json',release/'BASELINE_AOS_RELEASE_MANIFEST.json')
    manifest.update(policy_sha256=report['deployment_sha256'],checkpoint_sha256=report['checkpoint_sha256'],
        source_policy=str(export/'model.onnx'),source_machine='1516 via NX/AOS read-only',
        baseline_source_sha256=SOURCE_HASHES,native_rebuild_required=True,
        control_started_at_packaging=False,hardware_locomotion_validation='not_performed')
    (release/'AOS_RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    refresh_checksums(release)
    print('NATIVE_1516_PREPARATION=PASS no_joint_commands')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--export',type=Path,required=True)
    args=parser.parse_args();prepare(args.release.resolve(),args.export.resolve())
