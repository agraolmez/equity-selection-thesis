"""Build and atomically publish the immutable Stage 12 pre-final freeze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from portfolio_bot.data.secure_publication import (
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.final_freeze.inventory import (
    StableFile,
    Stage12InventoryError,
    authenticate_authority_inventory,
    build_environment_lock,
    build_project_file_inventory,
    canonical_json_bytes,
    confined_existing_path,
    confined_relative_path,
    load_json_object,
    require,
    sha256_bytes,
    stable_file,
)
from portfolio_bot.final_freeze.trial_ledger import (
    EXPECTED_RAW_TRIAL_COUNT,
    build_frozen_trial_ledger,
)
from portfolio_bot.final_freeze.regression import SANITIZED_ENVIRONMENT


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs/phase_06_final_test/stage_12_freeze_policy_v12.json"
)
BASE_POLICY = (
    PROJECT_ROOT / "configs/phase_06_final_test/stage_12_freeze_policy_v1.json"
)
MATERIALITY_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/stage_12_materiality_amendment_v1.json"
)
CALIBRATION_SCOPE_AMENDMENT = (
    PROJECT_ROOT
    / "configs/phase_06_final_test/stage_12_calibration_scope_amendment_v1.json"
)
POST_UNBLIND_TECHNICAL_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_nullability_transport_amendment_v1.json"
)
POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_scratch_topology_amendment_v1.json"
)
POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_zero_row_merge_provenance_amendment_v1.json"
)
STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_13_source_checkpoint_context_migration_v2.json"
)
POST_UNBLIND_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_december_market_input_scope_partition_amendment_v1.json"
)
STAGE13_DECISION_CHECKPOINT_REBIND = (
    PROJECT_ROOT
    / "configs/phase_06_final_test/stage_13_decision_checkpoint_rebind_v2.json"
)
POST_UNBLIND_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_entry_market_canonicalization_scope_alignment_"
    "amendment_v1.json"
)
POST_UNBLIND_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_accounting_missing_execution_cost_transport_"
    "amendment_v1.json"
)
POST_UNBLIND_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_logical_fingerprint_transport_amendment_v1.json"
)
POST_UNBLIND_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_factor_covariance_ordinal_basis_alignment_"
    "amendment_v1.json"
)
POST_UNBLIND_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_12_post_unblind_equal_weight_missing_cost_gate_amendment_v1.json"
)
STAGE13_PUBLIC_INTEREST_LAST_TRADABLE_CLOSE_LIQUIDATION = (
    PROJECT_ROOT / "configs/phase_06_final_test/"
    "stage_13_public_interest_last_tradable_close_liquidation_v1.json"
)
PRIMARY_FILES = (
    "_SUCCESS.json",
    "authority_inventory.json",
    "environment_lock.json",
    "file_inventory.json",
    "final_input_registry.json",
    "final_system.json",
    "freeze_checks.json",
    "pre_final_manifest.json",
    "replacement_inventory_delta.json",
    "source_drift.json",
    "test_receipts.json",
    "trial_ledger.json",
)
_EXPECTED_FINAL_SYSTEM_SHA256 = (
    "7fb0209dac9eee810bba3566071bae4a54c24ce9f80c74127455ed0bd980fe11"
)
_EXPECTED_POLICY_SHA256 = (
    "eff89acee0c6cdd85e4185ed2fb01718f839cb1a914e94c6fd3a5320363f83bf"
)
_EXPECTED_BASE_POLICY_SHA256 = (
    "68b47869472a51453856822283bc68e065640fb0c447d1a87f409ad9c422bf31"
)
_EXPECTED_POST_UNBLIND_TECHNICAL_AMENDMENT_SHA256 = (
    "6068df1a98ecdb47979a107e0370f0ac84c18b24a0a89dd69f9940d086991ca6"
)
_EXPECTED_PRIOR_REPLACEMENT_POLICY_SHA256 = (
    "e023704ac6eeb99c1476b22ce1b30cf7da655317ed9c5ff52b15f0fe8c371130"
)
_EXPECTED_PRIOR_V4_REPLACEMENT_POLICY_SHA256 = (
    "3bbb7eebbe200f477feaf688feb5a5faf1b92684c35d7ee1e70a2421df5481e3"
)
_EXPECTED_PRIOR_V5_REPLACEMENT_POLICY_SHA256 = (
    "be8486b694926758013a71bde959d24001d0d9b2d9513e94384ee69ccd74adfa"
)
_EXPECTED_PRIOR_V6_REPLACEMENT_POLICY_SHA256 = (
    "cb599fa425b6fe5c367d28be5c51b0b8d88241c7d1528a5da35632b010128f7b"
)
_EXPECTED_PRIOR_V7_REPLACEMENT_POLICY_SHA256 = (
    "c174b7eae80c672355172fd95ac3de1502b815160f0b24ecfcce7e2ede5c1e0f"
)
_EXPECTED_PRIOR_V8_REPLACEMENT_POLICY_SHA256 = (
    "991a46141e751217850dab3b87d540c1a0544082c241b70e1e6cbdfd7df9e5d9"
)
_EXPECTED_PRIOR_V9_REPLACEMENT_POLICY_SHA256 = (
    "ffc0aa63850635f6bd2004db7001d6623bc819ef52755ca33f62d0c13eee70a4"
)
_EXPECTED_PRIOR_V10_REPLACEMENT_POLICY_SHA256 = (
    "fb2d0f27020c1ec842fddbbb73bc6b1836088f5fef417a40974509d74827138f"
)
_EXPECTED_UNPUBLISHED_V11_POLICY_SHA256 = (
    "baa8ac787797a368dd84711603627e8aa3b5baf6cfd4b6c3ba022e27ab70dad4"
)
_EXPECTED_PRIOR_V2_REPLACEMENT_POLICY_SHA256 = (
    "99531170627ff7c28bb6f05d3e88aad94cef6d33f07f658283b2e2a800367394"
)
_EXPECTED_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_SHA256 = (
    "8e394a97d61e7f18b16baef65a29402960a73f83871151047012a3ee48ea97fb"
)
_EXPECTED_POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_SHA256 = (
    "ed224a251c99c6ec5c47a5856f872e052e903ac8959f50204636efe9e1abc7aa"
)
_EXPECTED_ZERO_ROW_MERGE_PROVENANCE_AUTHORIZATION_SHA256 = (
    "aa669c218dac6ab82df278f6a1013261b2f5d3f32335ac295f7cca225af40c4f"
)
_EXPECTED_STAGE13_ZERO_ROW_COPY_GUARD_BLOCKER_SHA256 = (
    "7fc5fecfefd4c291bc6196e16d7fd2f14f0763cf0cefcc1475ef869c44465bd7"
)
_EXPECTED_ZERO_ROW_ADAPTER_PRE_SHA256 = (
    "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
)
_EXPECTED_ZERO_ROW_ADAPTER_POST_SHA256 = (
    "a2d8ac04a84f957c61fca922b353116aa7c48512d188a2fb3f0547be5f1f6d45"
)
_EXPECTED_ZERO_ROW_REPLAY_PRE_SHA256 = (
    "4cd88a6fbb702207004f66006364efc5cf0a7d66633ebb6361bbb6f2cc2c53de"
)
_EXPECTED_ZERO_ROW_REPLAY_POST_SHA256 = (
    "ebecc9d341331cb7e6dc0b15cd45bc522ed7dda5796608542b53412c8ccc28c6"
)
_EXPECTED_SCRATCH_TOPOLOGY_AUTHORIZATION_SHA256 = (
    "3f0a285671144a91eccccf84cc345c62a73b21e379d9debf0d00c40a44fab3d2"
)
_EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256 = (
    "927fd04990a93ac01faf67bee9e600cf345d830e323c708420fd0d1ab1c6069e"
)
_EXPECTED_POST_UNBLIND_HASH_AUDIT_INCIDENT_SHA256 = (
    "e0b9ad71bb6a42f8bd0cefdcd08aaaa1e44d0cbb440814de80e626f678edbb36"
)
_EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V1_SHA256 = (
    "956fe984d350cff58142affb2f3832f0bd7b061749f61f998b2d817c59012b13"
)
_EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_SHA256 = (
    "d504e2dc75c58645b458d255db4f140bd0c9d8a9fa20022b021b4098f586613e"
)
_EXPECTED_PRE_AMENDMENT_ADAPTER_SHA256 = (
    "f02c0429a126a6a576197d2175030bc8a0018969ba01732b254b4a8318cfff31"
)
_EXPECTED_POST_AMENDMENT_ADAPTER_SHA256 = (
    "e2b3a85d32c87f0de42b67f75972a6a6203073fb603769cd7e11cbcfb69b7ec0"
)
_EXPECTED_CALIBRATION_SCOPE_AMENDMENT_SHA256 = (
    "6c44ab873f7c7c6ff69db4661a0edf1ad9d05f36ff9386f3eb96f6000febccf1"
)
_EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_SHA256 = (
    "302d4994195f77d2fe554e0411915a676e41a1ba1f2800634f805e344a9827a7"
)
_EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AUTHORIZATION_SHA256 = (
    "b626fba569d16a00ee0235006f1cd66cd9c8f6021c94686e854643411a03313a"
)
_EXPECTED_STAGE13_DECEMBER_MARKET_INPUT_BLOCKER_SHA256 = (
    "82a08d940c3b7f6f54bdbd4bc3e6690754f8298c70e4ce57ad366ea80528329b"
)
_EXPECTED_STAGE13_DECISION_CHECKPOINT_REBIND_SHA256 = (
    "1453e467c622913660a9097fc260c5632bbd963474f5b99c167478212ee70908"
)
_EXPECTED_STAGE13_DECISION_CHECKPOINT_REBIND_V1_SHA256 = (
    "178db1903c087cb53b5f6deb34b6df85da92ae66e55af2124ecf63a69f9fec27"
)
_EXPECTED_CHECKPOINT_REBIND_METADATA_AUTHORIZATION_SHA256 = (
    "206c3623db5c702f04aea4d6d9c8fd76225a29a9e20f8dca77cd8884cfb3f5b5"
)
_EXPECTED_CHECKPOINT_REBIND_METADATA_BLOCKER_SHA256 = (
    "f7ef225b6c6941b399d806b23525d50784b185d00b1ee178124165f763ec93f5"
)
_EXPECTED_CHECKPOINT_REBIND_METADATA_PREFLIGHT_SHA256 = (
    "b533cc91f12d8a92fdc9ac510d3f8b416b02e6827bbc5e1e8fc4d91af213acdd"
)
_EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AMENDMENT_SHA256 = (
    "fb01f060431c4b568053f1588b5c4d21e35371751762c87f76cc566dfa007626"
)
_EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AUTHORIZATION_SHA256 = (
    "56399168bfcf3b6476018cd0935f7e02d930294330d0e5c0f25a44bd7a75e820"
)
_EXPECTED_STAGE13_ENTRY_MARKET_PROJECTION_BLOCKER_SHA256 = (
    "80eec8044486f63b92b12f63de3a39d3ba2840561b0f318874963e1e2ec121ac"
)
_EXPECTED_STAGE13_ENTRY_MARKET_SHARE_PREFLIGHT_SHA256 = (
    "f53e09331597f8197eb38648ed280032ee89601b821e050cab930f0c3e3205a0"
)
_EXPECTED_ENTRY_MARKET_REPLAY_PRE_SHA256 = (
    "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
)
_EXPECTED_ENTRY_MARKET_REPLAY_POST_SHA256 = (
    "c6e439f3411aa95692e890be8e8364547b92368e19e8a4af9f7b08ba4d218d58"
)
_EXPECTED_CHECKPOINT_REBIND_REPLAY_POST_SHA256 = (
    "48f84559528749d8090858ad21de59f8a3e31a92aa6250a62ebfcb12a90b7253"
)
_EXPECTED_ACCOUNTING_TRANSPORT_AMENDMENT_SHA256 = (
    "2539bdc9b402be3df192690571e01b50ba83f90a381a9860066c62ee874d3b02"
)
_EXPECTED_ACCOUNTING_TRANSPORT_AUTHORIZATION_SHA256 = (
    "593c0da4c8a6997a06626b97fc51c3111ed470863394731bf4dac560c89bd80c"
)
_EXPECTED_THESIS_ERROR_MATERIALITY_POLICY_SHA256 = (
    "622d1335c540dfa84a4a396bade529e2e3632d1090a41ce66e1990dc582df03d"
)
_EXPECTED_STAGE13_ADV20_NULLABILITY_BLOCKER_SHA256 = (
    "9468526f52dae7af4b60589a1efda362c562f2e0a4e739c0fb2a22997f6dc5e3"
)
_EXPECTED_STAGE13_ACCOUNTING_PREFLIGHT_SHA256 = (
    "feb761454b5ca2ff2cd3b58faad4a902b67e3af57afcc25eed78eef081022deb"
)
_EXPECTED_ACCOUNTING_REPLAY_PRE_SHA256 = (
    "48f84559528749d8090858ad21de59f8a3e31a92aa6250a62ebfcb12a90b7253"
)
_EXPECTED_ACCOUNTING_REPLAY_POST_SHA256 = (
    "f88c9aa6532659c3d4b0cbddaa912a29d2d3d4594b01a5b184fa5edcfc99c664"
)
_EXPECTED_ACCOUNTING_MATERIALIZER_PRE_SHA256 = (
    "1e2f8d7ce33782daccb5d7697f64c52e081279dffb9610bc93f37485e5e47e18"
)
_EXPECTED_ACCOUNTING_MATERIALIZER_POST_SHA256 = (
    "23fcb0e5a0752ae376e6bc9584773051fe3e53d1992f430094b5aa1ab62dc925"
)
_EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SHA256 = (
    "b1fe0a1c9064bd26a025b21ac0a6d88d50a2050677a5779b25e822b68ab6a6c7"
)
_EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SEMANTIC_SHA256 = (
    "16a8ac6837d434849ca127fad4af9f1272d6483e6cf58aa2bfb8f8b26a4c5e6a"
)
_EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AUTHORIZATION_SHA256 = (
    "cbfa2ee2afd71294d71ea201c83c79a845ae4f87ace205a2f19e5c7d9dbac5e5"
)
_EXPECTED_STAGE13_LOGICAL_FINGERPRINT_BLOCKER_SHA256 = (
    "5b14591d9c559cb01f27262ceb7e7eaa12af2702e0575ccbf57b39a227c5d197"
)
_EXPECTED_STAGE13_LOGICAL_FINGERPRINT_PREFLIGHT_SHA256 = (
    "8daa9c7e078072bc9fcc73249086eeaf9c608192dd33331942b6896ea3013cf4"
)
_EXPECTED_LOGICAL_FINGERPRINT_REPLAY_PRE_SHA256 = (
    "f88c9aa6532659c3d4b0cbddaa912a29d2d3d4594b01a5b184fa5edcfc99c664"
)
_EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256 = (
    "25cea785f5eb3c390041221ee07d0333ca9f0374ebab5d4702ebd8e2302c1cfa"
)
_EXPECTED_ENTRY_MARKET_EXTENSION_PRE_SHA256 = (
    "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
)
_EXPECTED_ENTRY_MARKET_EXTENSION_POST_SHA256 = (
    "a5118f43874191e7d4ed3a6c0c2c5730dacb3052e72adbc1b97aee77952a8556"
)
_EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_SHA256 = (
    "9d1f8e93ef49b1621df442fd1760fe24fd6f405bcf06e7f1bd6d67b574f01f76"
)
_EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_SEMANTIC_SHA256 = (
    "b1365b6bc7206ceedf62c70ba28986207fcb5bfaa3ff05b71458d130316e168c"
)
_EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AUTHORIZATION_SHA256 = (
    "9c762291ce317c92837380ab02ccdf820d82c254f0c10252f5f3f715b40eb001"
)
_EXPECTED_STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_BLOCKER_SHA256 = (
    "3d8e1a80741aadae7773368d17bc04541e01b4aee21a48e8ebe2f3b278630a18"
)
_EXPECTED_STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_PREFLIGHT_SHA256 = (
    "49e043391a4ea27f8392a55685cfd72151d4ebd9c1119f7b2f05b698c6926530"
)
_EXPECTED_FAILED_STAGE13_V9_LAUNCH_MEMO_SHA256 = (
    "9c0c3d1b675e86b8dad97bfcb5361c1f7fae212e73710f8afe84448ac9d3db3e"
)
_EXPECTED_FACTOR_COVARIANCE_COMPILER_PRE_SHA256 = (
    "70dc0492e0538051a027c4f9b75bacebd307554cd16a383ba7143b00259ae082"
)
_EXPECTED_FACTOR_COVARIANCE_COMPILER_POST_SHA256 = (
    "21cc65d99c872c7fb6988da63412790d091c666b2caadd9b0859d21f718e1a8f"
)
_EXPECTED_FACTOR_COVARIANCE_RUNTIME_PRE_SHA256 = (
    "f36f07b00808eb4abcca5692c2b5732b67feaac15bb9369f36892cfdc4ef319b"
)
_EXPECTED_FACTOR_COVARIANCE_RUNTIME_POST_SHA256 = (
    "f7ed29de71bd2b3731bc9e3bc6385414ce264bebb499f9faa3a4ddef3657e929"
)
_EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SHA256 = (
    "72a94490bf1c835871104c966991ce42ef5a5ee5ee33ab7aabb20570d3de8ffa"
)
_EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SEMANTIC_SHA256 = (
    "4968272bb6117cb3c808e1f0679b1053aaa3b3a3459bf7c9e8105f7b03d00b7f"
)
_EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AUTHORIZATION_SHA256 = (
    "6f8063509a88792cb22d4d446a3633dc040a8a53030f6b79ae7e5076a8850a95"
)
_EXPECTED_STAGE13_RESUME3_FAILURE_SHA256 = (
    "d0e4b76bf16bc2caa56e218b8af7159abb9dfaa0992e9b0a14104a6239112056"
)
_EXPECTED_STAGE13_EQUAL_WEIGHT_MISSING_COST_PREFLIGHT_SHA256 = (
    "08c43ef17fad63878f3c996a8f863f75039ba707a849d6adc5235e30d9e1e5cf"
)
_EXPECTED_EQUAL_WEIGHT_KERNEL_PRE_SHA256 = (
    "e8698fce50883217f22c5a6404eacaa101eb750fb7092589a47e4cef8ba26ddf"
)
_EXPECTED_EQUAL_WEIGHT_KERNEL_POST_SHA256 = (
    "c0b57ad65e4c11b3836f195129f250d9e4c44be49e10d1bbf15b693f703937e0"
)
_EXPECTED_EQUAL_WEIGHT_PUBLICATION_PRE_SHA256 = (
    "7b20de47a4d04e67e42f9140d7c59e522f36470994f88c3bd5340f6a3f880755"
)
_EXPECTED_EQUAL_WEIGHT_PUBLICATION_POST_SHA256 = (
    "14982491210ef1412d0a5c04b051a829e5f547ca0a6f91cfe74ed64a4ce715d7"
)
_EXPECTED_EQUAL_WEIGHT_RUNTIME_PRE_SHA256 = (
    "f7ed29de71bd2b3731bc9e3bc6385414ce264bebb499f9faa3a4ddef3657e929"
)
_EXPECTED_EQUAL_WEIGHT_RUNTIME_POST_SHA256 = (
    "5390d06e5220579dc2045ab67031521a3c94e7603ac3e945eaeb80d78375d75a"
)
_EXPECTED_EQUAL_WEIGHT_TEST_PRE_SHA256 = (
    "9a9a1a7bbcbf3d1e7cb9f2e69b57c9416be29993a77a622d857594fd2ac9470e"
)
_EXPECTED_EQUAL_WEIGHT_TEST_POST_SHA256 = (
    "e0547f398444a20543b5784ca1716f271e78538c1f84a41443a6f20e88fb9217"
)
_EXPECTED_EQUAL_WEIGHT_PUBLICATION_TEST_PRE_SHA256 = (
    "782c498a2619a873f47f82b5e4c4011b5a6d38d78a97250f09048f4ead8e30c4"
)
_EXPECTED_EQUAL_WEIGHT_PUBLICATION_TEST_POST_SHA256 = (
    "97a4eeebd8900874eb75a572ede262ee03107e1d7e924e3c9da730b45d0e7cb7"
)
_EXPECTED_POLICY_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "stage",
        "status_before_publication",
        "methodology",
        "scope",
        "required_authorities",
        "inventory",
        "development_trial_source",
        "structured_final_system",
        "final_input_registry",
        "report_templates",
        "tests",
        "outputs",
        "memoization",
        "stage12_approval",
        "base_policy",
        "prior_replacement_policy",
        "post_unblind_technical_amendment",
        "post_unblind_scratch_topology_amendment",
        "post_unblind_zero_row_merge_provenance_amendment",
        "stage13_source_checkpoint_context_migration",
        "post_unblind_december_market_input_scope_partition_amendment",
        "stage13_decision_checkpoint_rebind",
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment",
        "post_unblind_accounting_missing_execution_cost_transport_amendment",
        "post_unblind_logical_fingerprint_transport_amendment",
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment",
        "post_unblind_equal_weight_missing_cost_gate_amendment",
        "unpublished_superseded_policy",
        "stage13_public_interest_last_tradable_close_liquidation",
        "consolidated_stage13_completion_governance",
    }
)
_EXPECTED_FINAL_INPUT_REGISTRATIONS = frozenset(
    {
        "final_calendar",
        "primary_predictors",
        "core_predictors",
        "partial_predictors",
        "primary_risk_cost",
        "core_risk_cost",
        "partial_risk_cost",
        "primary_factor_covariance",
        "core_factor_covariance",
        "partial_factor_covariance",
        "primary_risk_cost_month_summary",
        "core_risk_cost_month_summary",
        "partial_risk_cost_month_summary",
        "primary_cost_inputs",
        "core_cost_inputs",
        "partial_cost_inputs",
        "primary_market_inputs",
        "core_market_inputs",
        "partial_market_inputs",
        "primary_market_accounting",
        "core_market_accounting",
        "partial_market_accounting",
        "primary_factors",
        "core_factors",
        "primary_benchmarks",
        "core_benchmarks",
        "partial_benchmarks",
        "primary_completed_labels",
        "core_completed_labels",
        "primary_outcomes",
        "core_outcomes",
        "partial_descriptive_outcome",
    }
)
_SUPPORTED_ARROW_TYPES = frozenset(
    {
        "bool",
        "date32",
        "decimal128(38,1)",
        "decimal128(38,2)",
        "decimal128(38,6)",
        "decimal128(38,12)",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "list<string>",
        "string",
    }
)
_STAGE13_BLOCKERS: list[str] = []
_PRIOR_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-5fa8538a56b4ae01932f0705/file_inventory.json"
)
_PRIOR_STAGE12_FILE_INVENTORY_SHA256 = (
    "9e4f0f8d9662869ba4715b4d0474a1950ab7823ca4d48d057efd0f57f1874871"
)
_PRIOR_V4_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-8641d95070c80372cfa2ad02/file_inventory.json"
)
_PRIOR_V4_STAGE12_FILE_INVENTORY_SHA256 = (
    "87200b90aba352f761fddb61036f2888814067544d175b9855f64a6c727a6928"
)
_PRIOR_V5_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-fce138650a3857d90f0b65a2/file_inventory.json"
)
_PRIOR_V5_STAGE12_FILE_INVENTORY_SHA256 = (
    "182d0ae198340156a7e3c63310cade60de0a0ddb69e8e5488177f8aaa7c636dc"
)
_PRIOR_V6_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-3bcbc614d88b017b9c3979b0/file_inventory.json"
)
_PRIOR_V6_STAGE12_FILE_INVENTORY_SHA256 = (
    "27f85e75a476fc90404f81137fd7c7eb3960b8a073c1151b6292077890bd3a65"
)
_PRIOR_V7_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-58abf3e871270249636fd180/file_inventory.json"
)
_PRIOR_V7_STAGE12_FILE_INVENTORY_SHA256 = (
    "9721c5f0370b6397cc443499663a54a014ac755d03631e71067a660c888db3d7"
)
_PRIOR_V8_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-0e8c91f5940398fdf0fb9dba/file_inventory.json"
)
_PRIOR_V8_STAGE12_FILE_INVENTORY_SHA256 = (
    "36bc157b97089caa768f7e80fb677f3bbf97a6c3b91a9c96f0c0749a434538bf"
)
_PRIOR_V9_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-bdf5a063d5fee375556d3d88/file_inventory.json"
)
_PRIOR_V9_STAGE12_FILE_INVENTORY_SHA256 = (
    "ef709f1920723fe8deba82c55622f231fdfa9f9cd7780047abebadb2a6c151b5"
)
_PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH = (
    "artifacts/phase_06_final_test/stage_12_freezes/"
    "stage12-freeze-4159858fb5e321c1ec45c3fd/file_inventory.json"
)
_PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256 = (
    "ad81241b4b6449c9133b5799578c91abd9cd786ae7175345b1cd52d7209dd3ba"
)


class Stage12FreezeError(RuntimeError):
    """Raised when the pre-final freeze is incomplete or changed."""


@dataclass(frozen=True, slots=True)
class PreparedFreeze:
    policy: Mapping[str, Any]
    policy_sha256: str
    policy_path: Path
    documents: Mapping[str, Mapping[str, Any]]
    document_bytes: Mapping[str, bytes]
    document_receipts: Mapping[str, Mapping[str, Any]]
    freeze_id: str
    content_fingerprint: str


@dataclass(frozen=True, slots=True)
class PrimaryFreezePublication:
    freeze_id: str
    bundle: PublishedBundle
    pointer_path: Path
    pointer_sha256: str
    manifest_sha256: str


def _as_mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage12FreezeError(message)
    return value


def _is_consolidated_v12(policy: Mapping[str, Any]) -> bool:
    return policy.get("policy_id") == (
        "stage12-consolidated-stage13-completion-replacement-freeze-policy-v12"
    )


def _require_exact_mapping(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if dict(observed) != dict(expected):
        raise Stage12FreezeError(f"{label} differs")


def _load_policy(
    policy_path: Path,
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], str]:
    try:
        selected = confined_existing_path(
            policy_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 freeze policy",
        )
        policy, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("Stage 12 policy is invalid") from exc
    if (
        record.sha256 != _EXPECTED_POLICY_SHA256
        or set(policy) != _EXPECTED_POLICY_ROOT_KEYS
        or policy.get("schema_version") != 1
        or policy.get("stage") != 12
        or policy.get("policy_id")
        != ("stage12-consolidated-stage13-completion-replacement-freeze-policy-v12")
        or policy.get("status_before_publication")
        != (
            "approved_for_consolidated_substantive_economic_post_unblind_"
            "replacement_freeze"
        )
    ):
        raise Stage12FreezeError("Stage 12 policy identity differs")
    scope = _as_mapping(policy.get("scope"), "Stage 12 scope is missing")
    required_scope = {
        "purpose": (
            "refreeze_the_complete_v1_final_system_after_consolidating_all_"
            "authorized_stage13_corrections_and_the_substantive_2021_public_"
            "interest_last_tradable_close_liquidation_treatment_"
            "without_opening_parent_or_outcome_payloads"
        ),
        "repository_metadata_reads_permitted": True,
        "authenticated_development_trial_metadata_reads_permitted": True,
        "stage11_verified_metadata_reads_permitted": True,
        "raw_payload_file_opens_permitted": False,
        "final_period_payload_file_opens_permitted": False,
        "final_period_rows_read_required": 0,
        "final_forecasts_weights_trades_pnl_or_performance_created_required": 0,
        "stage13_execution_authorized_now": False,
        "stage13_requires_separate_user_approval": True,
    }
    _require_exact_mapping(scope, required_scope, label="Stage 12 scope")
    outputs = _as_mapping(policy.get("outputs"), "Stage 12 outputs are missing")
    _require_exact_mapping(
        outputs,
        {
            "artifact_root": "artifacts/phase_06_final_test",
            "primary_namespace": "stage_12_freezes",
            "verification_namespace": "stage_12_verifications",
            "checkpoint_namespace": "stage_12_checkpoints",
            "latest_primary_pointer": (
                "artifacts/phase_06_final_test/LATEST_STAGE_12_PRIMARY.json"
            ),
            "latest_verified_pointer": (
                "artifacts/phase_06_final_test/LATEST_STAGE_12_VERIFIED.json"
            ),
            "primary_status": (
                "STAGE12_POST_UNBLIND_REPLACEMENT_FREEZE_COMPLETE_PENDING_"
                "INDEPENDENT_VERIFICATION"
            ),
            "verified_status": "STAGE12_POST_UNBLIND_REPLACEMENT_FREEZE_VERIFIED",
        },
        label="Stage 12 output contract",
    )
    if policy.get("structured_final_system") != {
        "path": "configs/phase_06_final_test/stage_12_final_system_v1.json"
    } or policy.get("final_input_registry") != {
        "path": "configs/phase_06_final_test/stage_13_input_registry_v1.json"
    }:
        raise Stage12FreezeError("Stage 12 structured input paths differ")
    if policy.get("report_templates") != [
        "configs/phase_06_final_test/stage_13_report_template_v1.md",
        "configs/phase_06_final_test/stage_13_report_template_v2.md",
        "configs/phase_06_final_test/stage_14_report_template_v1.md",
    ]:
        raise Stage12FreezeError("Stage 12 report template paths differ")
    _require_exact_mapping(
        _as_mapping(policy.get("base_policy"), "base Stage 12 policy is missing"),
        {
            "path": "configs/phase_06_final_test/stage_12_freeze_policy_v1.json",
            "sha256": _EXPECTED_BASE_POLICY_SHA256,
        },
        label="base Stage 12 policy binding",
    )
    try:
        base_policy_path = confined_relative_path(
            project_root,
            policy["base_policy"]["path"],
            file_required=True,
            label="base Stage 12 policy",
        )
        base_policy_record = stable_file(base_policy_path, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("base Stage 12 policy path is unsafe") from exc
    if base_policy_record.sha256 != _EXPECTED_BASE_POLICY_SHA256:
        raise Stage12FreezeError("base Stage 12 policy SHA-256 differs")
    _require_exact_mapping(
        _as_mapping(
            policy.get("prior_replacement_policy"),
            "prior replacement Stage 12 policy is missing",
        ),
        {
            "path": "configs/phase_06_final_test/stage_12_freeze_policy_v10.json",
            "sha256": _EXPECTED_PRIOR_V10_REPLACEMENT_POLICY_SHA256,
        },
        label="prior replacement Stage 12 policy binding",
    )
    try:
        prior_policy_path = confined_relative_path(
            project_root,
            policy["prior_replacement_policy"]["path"],
            file_required=True,
            label="prior replacement Stage 12 policy",
        )
        prior_policy_record = stable_file(prior_policy_path, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "prior replacement Stage 12 policy path is unsafe"
        ) from exc
    if prior_policy_record.sha256 != _EXPECTED_PRIOR_V10_REPLACEMENT_POLICY_SHA256:
        raise Stage12FreezeError("prior replacement Stage 12 policy SHA-256 differs")
    _require_exact_mapping(
        _as_mapping(
            policy.get("unpublished_superseded_policy"),
            "unpublished Stage 12 v11 policy record is missing",
        ),
        {
            "path": "configs/phase_06_final_test/stage_12_freeze_policy_v11.json",
            "sha256": _EXPECTED_UNPUBLISHED_V11_POLICY_SHA256,
            "status": "never_published_superseded_by_stage12_v12",
            "authoritative_stage12_identity": False,
        },
        label="unpublished Stage 12 v11 policy record",
    )
    terminal_policy_binding = _as_mapping(
        policy.get("stage13_public_interest_last_tradable_close_liquidation"),
        "Stage 13 terminal-treatment policy binding is missing",
    )
    if (
        set(terminal_policy_binding) != {"path", "sha256"}
        or terminal_policy_binding.get("path")
        != (
            "configs/phase_06_final_test/"
            "stage_13_public_interest_last_tradable_close_liquidation_v1.json"
        )
        or not isinstance(terminal_policy_binding.get("sha256"), str)
        or len(terminal_policy_binding["sha256"]) != 64
    ):
        raise Stage12FreezeError("Stage 13 terminal-treatment policy binding differs")
    terminal_policy_sha256 = str(terminal_policy_binding["sha256"])
    for relative, expected_sha, label in (
        (
            policy["unpublished_superseded_policy"]["path"],
            _EXPECTED_UNPUBLISHED_V11_POLICY_SHA256,
            "unpublished Stage 12 v11 policy",
        ),
        (
            policy["stage13_public_interest_last_tradable_close_liquidation"]["path"],
            terminal_policy_sha256,
            "Stage 13 terminal-treatment policy",
        ),
    ):
        try:
            selected_bound_path = confined_relative_path(
                project_root,
                relative,
                file_required=True,
                label=label,
            )
            selected_bound_record = stable_file(
                selected_bound_path,
                trusted_root=project_root,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if selected_bound_record.sha256 != expected_sha:
            raise Stage12FreezeError(f"{label} SHA-256 differs")
    memoization = _as_mapping(
        policy.get("memoization"), "Stage 12 memoization policy is missing"
    )
    if (
        memoization.get("resume_rule")
        != "reuse_only_after_hash_schema_and_semantic_reauthentication"
        or memoization.get("recompute_rule")
        != "recompute_only_the_failed_node_and_its_descendants"
        or memoization.get("completed_stage11_may_be_replayed") is not False
    ):
        raise Stage12FreezeError("Stage 12 memoization boundary differs")
    return policy, record.sha256


def _validate_methodology(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    methodology = _as_mapping(
        policy.get("methodology"), "methodology binding is missing"
    )
    records: dict[str, Any] = {}
    for role, path_key, sha_key in (
        ("source", "source", "source_sha256"),
        ("freeze_receipt", "freeze_receipt", "freeze_receipt_sha256"),
    ):
        relative = str(methodology.get(path_key))
        try:
            selected = confined_relative_path(
                project_root,
                relative,
                file_required=True,
                label=f"methodology {role}",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"unsafe methodology {role} path") from exc
        if record.sha256 != methodology.get(sha_key):
            raise Stage12FreezeError(f"methodology {role} SHA-256 differs")
        records[role] = {
            "relative_path": relative,
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }
    return records


def _validate_calibration_scope_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    relative = (
        "configs/phase_06_final_test/" "stage_12_calibration_scope_amendment_v1.json"
    )
    required = policy.get("required_authorities")
    authority = (
        [
            item
            for item in required
            if isinstance(item, Mapping)
            and item.get("role") == "pre_unblind_calibration_scope_amendment"
        ]
        if isinstance(required, list)
        else []
    )
    if authority != [
        {
            "role": "pre_unblind_calibration_scope_amendment",
            "path": relative,
            "sha256": _EXPECTED_CALIBRATION_SCOPE_AMENDMENT_SHA256,
        }
    ]:
        raise Stage12FreezeError(
            "pre-unblind calibration-scope amendment authority differs"
        )
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="pre-unblind calibration-scope amendment",
        )
        amendment, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "pre-unblind calibration-scope amendment path is unsafe"
        ) from exc
    governance = _as_mapping(
        amendment.get("governance"),
        "calibration-scope amendment governance is missing",
    )
    base = _as_mapping(
        amendment.get("base_methodology_authority"),
        "calibration-scope amendment base authority is missing",
    )
    conflict = _as_mapping(
        amendment.get("explicit_methodology_conflict"),
        "calibration-scope amendment conflict is missing",
    )
    rationale = _as_mapping(
        amendment.get("scientific_rationale"),
        "calibration-scope amendment rationale is missing",
    )
    execution = _as_mapping(
        amendment.get("execution_contract"),
        "calibration-scope amendment execution contract is missing",
    )
    economic = _as_mapping(
        amendment.get("economic_effect"),
        "calibration-scope amendment economic effect is missing",
    )
    access = _as_mapping(
        amendment.get("timing_and_access_boundary"),
        "calibration-scope amendment access boundary is missing",
    )
    if (
        record.sha256 != _EXPECTED_CALIBRATION_SCOPE_AMENDMENT_SHA256
        or set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "governance",
            "base_methodology_authority",
            "explicit_methodology_conflict",
            "scientific_rationale",
            "execution_contract",
            "economic_effect",
            "timing_and_access_boundary",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-prefinal-shared-calibration-scope-v1"
        or amendment.get("status") != "STAGE12_PREFINAL_METHODOLOGY_AMENDMENT_FROZEN"
        or governance
        != {
            "provenance": "agent_authored_prefinal_implementation_amendment",
            "explicit_user_acknowledgement": False,
            "governing_user_scope": ("complete_stage12_without_final_period_access"),
            "approval_receipt_binding_asserts_user_acceptance": False,
        }
        or base
        != {
            "methodology_version": "v1",
            "source_path": "outputs/methodology_v1_source.md",
            "source_sha256": (
                "77ac9a4a79b0f46188b56f9a9a3d4affccdbfb13feafdc0f5facc79f2bbc6e81"
            ),
            "freeze_receipt_path": "outputs/methodology_v1_freeze.json",
            "freeze_receipt_sha256": (
                "cbd897684639a2ccfe1ecf266cf458ef836017b91310fed2e266a75c38808d98"
            ),
        }
        or conflict.get("source_section")
        != "Ensemble, alpha calibration, and refitting"
        or "separately refit and calibrated" not in str(conflict.get("frozen_v1_rule"))
        or "shared byte-identically" not in str(conflict.get("amended_rule"))
        or rationale.get("authenticated_prefinal_reduced_core_oof_receipts_exist")
        is not False
        or rationale.get("authenticated_prefinal_full_feature_oof_receipts_exist")
        is not True
        or rationale.get("calibration_source_scope") != "full_feature_primary"
        or rationale.get("reduced_core_scores_relabelled_as_prefinal_oof") is not False
        or rationale.get("post_selection_reduced_core_oof_history_manufactured")
        is not False
        or rationale.get(
            "same_strategy_calibration_receipt_shared_across_feature_scopes"
        )
        is not True
        or rationale.get("separate_full_and_core_model_parameter_refits_retained")
        is not True
        or rationale.get("quarterly_updates_use_completed_labels_only") is not True
        or execution
        != {
            "calibrated_strategy_ids": [
                "ensemble_hybrid_primary",
                "elastic_net_hybrid_challenger",
                "fixed_value_quality_momentum",
            ],
            "quarterly_fit_signal_months": [3, 6, 9, 12],
            "calibration_history_sources": [
                "full_feature_prefinal_pseudo_oof",
                "full_feature_final_prior_live",
            ],
            "completed_label_cutoff": {
                "holding_month_strictly_before_fit_holding_month": True,
                "exit_date_on_or_before_fit_signal_date": True,
                "outcome_availability_date_on_or_before_fit_signal_date": True,
            },
            "identical_receipt_and_sha256_stored_under_full_and_core_scope_keys": True,
        }
        or economic
        != {
            "economic_behavior_changed": True,
            "primary_full_feature_calibration_behavior_changed": False,
            "reduced_core_calibration_behavior_changed": True,
            "primary_and_core_pnl_may_be_spliced": False,
            "post_unblind_policy_change_permitted": False,
        }
        or access
        != {
            "frozen_before_final_period_access": True,
            "final_period_payload_files_opened": 0,
            "final_period_rows_read": 0,
            "final_forecasts_created": 0,
            "final_weights_created": 0,
            "final_trades_created": 0,
            "final_pnl_or_performance_rows_created": 0,
        }
    ):
        raise Stage12FreezeError(
            "pre-unblind calibration-scope amendment contract differs"
        )
    return amendment, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "amendment_id": str(amendment["amendment_id"]),
        "status": str(amendment["status"]),
        "timing_and_access_boundary": dict(access),
    }


def _authenticate_post_unblind_hash_audit_incident(
    binding: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Stable-open and semantically authenticate the disclosed hash-only access."""

    expected_binding = {
        "path": (
            "artifacts/phase_06_final_test/"
            "STAGE_12_POST_UNBLIND_HASH_AUDIT_INCIDENT.json"
        ),
        "sha256": _EXPECTED_POST_UNBLIND_HASH_AUDIT_INCIDENT_SHA256,
        "status": "STAGE12_POST_UNBLIND_HASH_AUDIT_INCIDENT_DISCLOSED",
        "result_informed_choice": False,
        "guarded_stage12_build_or_test_execution": False,
        "must_be_independently_reverified": True,
    }
    _require_exact_mapping(
        binding,
        expected_binding,
        label="post-unblind hash-audit incident binding",
    )
    required = policy.get("required_authorities")
    incident_authorities = (
        [
            item
            for item in required
            if isinstance(item, Mapping)
            and item.get("role") == "post_unblind_hash_audit_incident"
        ]
        if isinstance(required, list)
        else []
    )
    if incident_authorities != [
        {
            "role": "post_unblind_hash_audit_incident",
            "path": expected_binding["path"],
            "sha256": expected_binding["sha256"],
        }
    ]:
        raise Stage12FreezeError("post-unblind hash-audit incident authority differs")
    try:
        selected = confined_relative_path(
            project_root,
            expected_binding["path"],
            file_required=True,
            label="post-unblind hash-audit incident",
        )
        incident, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "post-unblind hash-audit incident path is unsafe"
        ) from exc
    date = _as_mapping(
        incident.get("date"), "post-unblind hash-audit incident date is missing"
    )
    context = _as_mapping(
        incident.get("access_context"),
        "post-unblind hash-audit incident context is missing",
    )
    facts = _as_mapping(
        incident.get("access_facts"),
        "post-unblind hash-audit incident facts are missing",
    )
    scientific = _as_mapping(
        incident.get("scientific_effect"),
        "post-unblind hash-audit incident scientific boundary is missing",
    )
    remediation = _as_mapping(
        incident.get("remediation_and_claim_boundary"),
        "post-unblind hash-audit incident remediation is missing",
    )
    if (
        record.sha256 != _EXPECTED_POST_UNBLIND_HASH_AUDIT_INCIDENT_SHA256
        or set(incident)
        != {
            "schema_version",
            "stage",
            "status",
            "incident_id",
            "date",
            "access_context",
            "access_facts",
            "scan_algorithms_attempted",
            "scientific_effect",
            "remediation_and_claim_boundary",
        }
        or incident.get("schema_version") != 1
        or incident.get("stage") != 12
        or incident.get("status")
        != "STAGE12_POST_UNBLIND_HASH_AUDIT_INCIDENT_DISCLOSED"
        or incident.get("incident_id")
        != "stage12-post-unblind-parent-prerequisite-hash-audit-incident-v1"
        or date.get("local_date") != "2026-07-26"
        or date.get("timezone") != "Europe/Istanbul"
        or date.get("exact_start_time_recorded") is not False
        or date.get("exact_end_time_recorded") is not False
        or context.get("stage12_guarded_build_or_test_execution") is not False
        or context.get("stage12_safe_runner_used") is not False
        or context.get("stage12_test_or_build_receipt_produced") is not False
        or context.get("parent_context_sha256")
        != "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5"
        or facts
        != {
            "complete_hash_scans": 5,
            "unique_predictor_prerequisite_payload_files": 64,
            "predictor_prerequisite_payload_file_open_events": 320,
            "bytes_read_per_complete_scan": 7190634134,
            "total_predictor_prerequisite_bytes_hashed": 35953170670,
            "outcome_or_economic_payload_files_opened": 0,
            "outcome_or_economic_payload_bytes_read": 0,
            "rows_parsed": 0,
            "values_parsed": 0,
            "rows_rendered": 0,
            "values_rendered": 0,
            "payload_content_rendered": False,
            "files_written": 0,
            "files_mutated": 0,
            "directories_mutated": 0,
        }
        or incident.get("scan_algorithms_attempted")
        != [
            "canonical_JSON_file_record_list_and_wrapper_variants",
            "record_key_wrapper_and_text_stream_variants",
            "path_prefix_record_key_and_wrapper_variants",
            "relative_path_NUL_binary_SHA256_and_size_stream_variants",
            "shasum_text_output_stream_variants",
        ]
        or scientific.get("scientific_or_economic_result_informed_choice") is not False
        or scientific.get(
            "integrity_implementation_choice_from_aggregate_digest_nonmatch"
        )
        is not True
        or scientific.get("data_rows_or_values_inspected") is not False
        or scientific.get("outcomes_or_economics_inspected") is not False
        or scientific.get("formulas_types_keys_linkage_rules_or_economics_changed")
        is not False
        or scientific.get("attempted_digest_outputs_used_as_authority") is not False
        or remediation.get(
            "further_parent_payload_reads_before_renewed_stage13_approval_forbidden"
        )
        is not True
        or remediation.get(
            "runtime_parent_payload_reauthentication_deferred_until_renewed_stage13_approval"
        )
        is not True
        or remediation.get(
            "stage12_build_and_test_zero_access_counters_apply_only_to_guarded_build_and_test_execution"
        )
        is not True
        or remediation.get("all_amendment_activity_had_zero_payload_access")
        is not False
        or remediation.get("incident_must_be_bound_by_replacement_stage12_freeze")
        is not True
        or remediation.get("incident_must_be_independently_reverified") is not True
    ):
        raise Stage12FreezeError("post-unblind hash-audit incident contract differs")
    return {
        "relative_path": str(expected_binding["path"]),
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "status": str(incident["status"]),
        "access_facts": dict(facts),
        "scientific_effect": dict(scientific),
        "remediation_and_claim_boundary": dict(remediation),
    }


def _authenticate_prior_replaced_authorities(
    historical: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Authenticate immutable copies of the three superseded authorities."""

    specifications = {
        "primary_pointer": (
            "prior_stage12_primary_pointer_path",
            "prior_stage12_primary_pointer_sha256",
        ),
        "verified_pointer": (
            "prior_stage12_verified_pointer_path",
            "prior_stage12_verified_pointer_sha256",
        ),
        "stage13_approval": (
            "prior_stage13_approval_path",
            "prior_stage13_approval_sha256",
        ),
        "stage13_resume_blocker": (
            "stage13_resume_blocker_path",
            "stage13_resume_blocker_sha256",
        ),
    }
    documents: dict[str, Mapping[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for role, (path_key, sha_key) in specifications.items():
        try:
            selected = confined_relative_path(
                project_root,
                historical[path_key],
                file_required=True,
                label=f"prior immutable {role}",
            )
            value, record = load_json_object(selected, trusted_root=project_root)
        except (KeyError, Stage12InventoryError) as exc:
            raise Stage12FreezeError(f"prior immutable {role} path is unsafe") from exc
        if record.sha256 != historical.get(sha_key):
            raise Stage12FreezeError(f"prior immutable {role} SHA-256 differs")
        documents[role] = value
        records[role] = {
            "relative_path": str(historical[path_key]),
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }
    primary = documents["primary_pointer"]
    verified = documents["verified_pointer"]
    approval = documents["stage13_approval"]
    blocker = documents["stage13_resume_blocker"]
    freeze_id = "stage12-freeze-78e7f9e6e9856597084a8046"
    manifest_sha = "6c26845811b5c089a123a905842f0d0e4597fb0204210b929bf4a040457b0110"
    if (
        primary.get("schema_version") != 1
        or primary.get("stage") != 12
        or primary.get("freeze_id") != freeze_id
        or primary.get("manifest_sha256") != manifest_sha
        or primary.get("status")
        != "STAGE12_PRE_FINAL_FREEZE_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
        or primary.get("bundle_path")
        != f"artifacts/phase_06_final_test/stage_12_freezes/{freeze_id}"
        or verified.get("schema_version") != 1
        or verified.get("stage") != 12
        or verified.get("primary_freeze_id") != freeze_id
        or verified.get("primary_manifest_sha256") != manifest_sha
        or verified.get("primary_pointer_sha256")
        != records["primary_pointer"]["sha256"]
        or verified.get("verification_id") != "stage12-verify-8e1241681dfe6eb3d4a5ce13"
        or verified.get("status") != "STAGE12_PRE_FINAL_FREEZE_VERIFIED"
        or verified.get("difference_count") != 0
        or approval.get("schema_version") != 1
        or approval.get("approved") is not True
        or approval.get("stage12_build_id") != freeze_id
        or approval.get("stage12_manifest_sha256") != manifest_sha
        or approval.get("stage12_verification_id") != verified.get("verification_id")
        or approval.get("stage12_verified_pointer_sha256")
        != records["verified_pointer"]["sha256"]
        or approval.get("stage12_binding_sha256")
        != "f9dbccbd448e5e36e52c83d43889549e7e12d4121d12ad049bc6a89962edc482"
        or approval.get("payload_sha256")
        != "256447acc775c383bb250c0ad0448a79696b51aee240e9417ec95311a8fc3a20"
        or blocker.get("schema_version") != 1
        or blocker.get("stage") != 13
        or blocker.get("status")
        != "FAILED_CLOSED_PENDING_EXPLICIT_TECHNICAL_AMENDMENT_AUTHORIZATION"
        or blocker.get("failed_run", {}).get("error_message")
        != "Development/extension schema differs: ccm_link_interval.parquet"
        or blocker.get("failed_run", {}).get("economic_or_outcome_months_executed") != 0
        or blocker.get("failed_run", {}).get("stage13_publication_pointers_created")
        != 0
        or blocker.get("root_cause", {}).get("affected_merge_table_count") != 14
        or blocker.get("root_cause", {}).get("economic_values_or_rules_differ")
        is not False
        or blocker.get("frozen_authority", {}).get("adapter_sha256")
        != _EXPECTED_PRE_AMENDMENT_ADAPTER_SHA256
        or blocker.get("frozen_authority", {}).get(
            "current_stage12_verified_pointer_sha256"
        )
        != records["verified_pointer"]["sha256"]
        or blocker.get("frozen_authority", {}).get(
            "current_stage13_approval_receipt_sha256"
        )
        != records["stage13_approval"]["sha256"]
    ):
        raise Stage12FreezeError("prior immutable authority semantics differ")
    approval_payload = dict(approval)
    approval_payload_sha = approval_payload.pop("payload_sha256", None)
    if sha256_bytes(canonical_json_bytes(approval_payload)) != approval_payload_sha:
        raise Stage12FreezeError("prior Stage 13 approval payload SHA-256 differs")
    for role, relative, expected_sha in (
        (
            "primary_manifest",
            f"{primary['bundle_path']}/pre_final_manifest.json",
            manifest_sha,
        ),
        (
            "verification_report",
            f"{verified['verification_bundle_path']}/verification_report.json",
            verified.get("verification_report_sha256"),
        ),
    ):
        try:
            selected = confined_relative_path(
                project_root,
                relative,
                file_required=True,
                label=f"prior immutable {role}",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"prior immutable {role} path is unsafe") from exc
        if record.sha256 != expected_sha:
            raise Stage12FreezeError(f"prior immutable {role} SHA-256 differs")
        records[role] = {
            "relative_path": relative,
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }
    return {
        "status": "PRIOR_REPLACED_AUTHORITIES_AUTHENTICATED",
        "records": records,
    }


def _validate_post_unblind_technical_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the one authorized non-economic post-unblind transition."""

    relative = (
        "configs/phase_06_final_test/"
        "stage_12_post_unblind_nullability_transport_amendment_v1.json"
    )
    expected_binding = {
        "path": relative,
        "sha256": _EXPECTED_POST_UNBLIND_TECHNICAL_AMENDMENT_SHA256,
    }
    _require_exact_mapping(
        _as_mapping(
            policy.get("post_unblind_technical_amendment"),
            "post-unblind technical amendment binding is missing",
        ),
        expected_binding,
        label="post-unblind technical amendment binding",
    )
    required = policy.get("required_authorities")
    authority = (
        [
            item
            for item in required
            if isinstance(item, Mapping)
            and item.get("role") == "post_unblind_technical_amendment"
        ]
        if isinstance(required, list)
        else []
    )
    if authority != [
        {
            "role": "post_unblind_technical_amendment",
            **expected_binding,
        }
    ]:
        raise Stage12FreezeError("post-unblind technical amendment authority differs")
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="post-unblind technical amendment",
        )
        amendment, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "post-unblind technical amendment path is unsafe"
        ) from exc
    authorization = _as_mapping(
        amendment.get("authorization"),
        "post-unblind amendment authorization is missing",
    )
    historical = _as_mapping(
        amendment.get("historical_bindings"),
        "post-unblind amendment historical bindings are missing",
    )
    transition = _as_mapping(
        amendment.get("source_transition"),
        "post-unblind amendment source transition is missing",
    )
    transport = _as_mapping(
        amendment.get("schema_transport_contract"),
        "post-unblind amendment schema contract is missing",
    )
    diagnosis = _as_mapping(
        amendment.get("diagnostic_evidence"),
        "post-unblind amendment diagnostic evidence is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "post-unblind amendment economic boundary is missing",
    )
    hash_audit_incident = _as_mapping(
        amendment.get("hash_audit_incident"),
        "post-unblind hash-audit incident binding is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "post-unblind amendment replacement authority is missing",
    )
    tests = _as_mapping(
        amendment.get("required_tests"),
        "post-unblind amendment test contract is missing",
    )
    if (
        record.sha256 != _EXPECTED_POST_UNBLIND_TECHNICAL_AMENDMENT_SHA256
        or set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization",
            "historical_bindings",
            "source_transition",
            "schema_transport_contract",
            "diagnostic_evidence",
            "non_economic_boundary",
            "hash_audit_incident",
            "replacement_authority",
            "stage13_source_checkpoint_context_migration",
            "required_tests",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-auxiliary-nullability-transport-v1"
        or amendment.get("status")
        != "STAGE12_POST_UNBLIND_TECHNICAL_AMENDMENT_AUTHORIZED"
        or amendment.get("classification")
        != "non_economic_rule_restoring_infrastructure_compatibility_correction"
        or authorization
        != {
            "source": "current_codex_task_user_message",
            "authorization_form": "contextual_affirmative_reply",
            "scope_request_authored_by_user": False,
            "scope_request_text": (
                "I authorize the narrowly scoped post-unblind technical amendment "
                "in STAGE_13_RESUME_BLOCKER.json, with zero-null proof and no "
                "changes to data, formulas, types, keys, linkage rules, or "
                "economics. Implement it, test it, and publish an independently "
                "verified replacement Stage 12 freeze."
            ),
            "scope_request_text_sha256": (
                "92887ebc5d93e83b721d46c0e6252330848dad367458bf8e48b4ad7d8ec3f1fe"
            ),
            "source_message_text": "ok",
            "source_message_text_sha256": (
                "2689367b205c16ce32ed4200942b8b8b1e262dfc70d9bc9fbc77c49699a4f1df"
            ),
            "authorized": True,
            "authorized_scope": (
                "only_the_blocker_described_narrow_arrow_nullability_transport_"
                "correction_and_replacement_stage12_refreeze"
            ),
            "replacement_stage13_approval_granted": False,
        }
        or historical
        != {
            "base_stage12_policy_path": (
                "configs/phase_06_final_test/stage_12_freeze_policy_v1.json"
            ),
            "base_stage12_policy_sha256": _EXPECTED_BASE_POLICY_SHA256,
            "prior_stage12_primary_pointer_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_PRIMARY_PRIOR_FREEZE_78E7F9E6.json"
            ),
            "prior_stage12_primary_pointer_sha256": (
                "7ebd59fe8d803cd7da06b0da8a1400f061bf043bf196f37d4a5639f0b5ca5735"
            ),
            "prior_stage12_verified_pointer_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_VERIFIED_PRIOR_FREEZE_78E7F9E6.json"
            ),
            "prior_stage12_verified_pointer_sha256": (
                "75c45ed9d6c3c6f326d8d148224c3036fb1f331f734bf3ae3d4245c7be5df75e"
            ),
            "prior_stage12_freeze_id": ("stage12-freeze-78e7f9e6e9856597084a8046"),
            "prior_stage12_manifest_sha256": (
                "6c26845811b5c089a123a905842f0d0e4597fb0204210b929bf4a040457b0110"
            ),
            "prior_stage12_verification_id": (
                "stage12-verify-8e1241681dfe6eb3d4a5ce13"
            ),
            "prior_stage13_approval_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_APPROVAL_PRIOR_STAGE12_FREEZE_78E7F9E6.json"
            ),
            "prior_stage13_approval_sha256": (
                "dab8e1465dc95eb46323ada79c7978151317fc55eacd35d95c3ae2325667049d"
            ),
            "stage13_resume_blocker_path": (
                "artifacts/phase_06_final_test/STAGE_13_RESUME_BLOCKER.json"
            ),
            "stage13_resume_blocker_sha256": (
                "d6c212f5272adcf2396f64a4483ce74de3ae74a0eed17c110ab569fc03c852a1"
            ),
        }
        or transition
        != {
            "adapter_path": (
                "src/portfolio_bot/validation/validation_feature_input_adapter.py"
            ),
            "pre_amendment_sha256": _EXPECTED_PRE_AMENDMENT_ADAPTER_SHA256,
            "post_amendment_sha256": _EXPECTED_POST_AMENDMENT_ADAPTER_SHA256,
            "exact_changed_stage11_bound_member_count": 1,
            "permitted_changed_stage11_bound_members": [
                "src/portfolio_bot/validation/validation_feature_input_adapter.py"
            ],
            "frozen_auxiliary_schema_path": (
                "configs/phase_02_canonicalization/auxiliary_source_schema_v1.json"
            ),
            "frozen_auxiliary_schema_sha256": (
                "98ea793e43799d17ce6914e41874a87fe962a09bb251e7b65d7f062cae4e808e"
            ),
        }
        or transport
        != {
            "affected_merge_table_count": 14,
            "ordered_field_names_must_match": True,
            "physical_arrow_types_must_match": True,
            "extension_schema_must_equal_frozen_phase2_schema": True,
            "permitted_nullability_transition": (
                "development_nullable_to_extension_nonnullable_only"
            ),
            "inverse_nullability_relaxation_permitted": False,
            "type_order_or_field_drift_permitted": False,
            "every_widened_development_required_column_zero_null_required": True,
            "per_column_zero_null_receipt_required": True,
            "row_value_date_key_precedence_sort_or_cardinality_change_permitted": False,
        }
        or diagnosis
        != {
            "failure_class": (
                "Development/extension schema differs: ccm_link_interval.parquet"
            ),
            "economic_or_outcome_months_executed_before_failure": 0,
            "stage13_publication_pointers_created_before_failure": 0,
            "observed_affected_merge_tables": 14,
            "observed_widened_required_field_pairs": 106,
            "observed_development_rows_covered_by_zero_null_proofs": 317047702,
            "observed_required_field_null_rows": 0,
            "result_informed_choice": False,
        }
        or boundary
        != {
            "scientific_formula_changed": False,
            "model_selection_or_frozen_economic_policy_changed": False,
            "source_rows_values_dates_keys_or_linkage_rules_changed": False,
            "stage11_recomputed": False,
            "final_period_predictor_prerequisite_unique_payload_files_opened_by_amendment_work": 64,
            "final_period_predictor_prerequisite_payload_file_open_events_by_amendment_work": 320,
            "final_period_predictor_prerequisite_bytes_hashed_by_amendment_work": 35953170670,
            "outcome_or_economic_payload_files_opened_by_amendment_work": 0,
            "final_period_rows_read_by_amendment_work": 0,
            "economic_months_executed_by_amendment_work": 0,
            "outcome_months_executed_by_amendment_work": 0,
            "final_forecasts_weights_trades_or_pnl_created_by_amendment_work": 0,
        }
        or replacement
        != {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_explicit_stage13_approval_bound_to_exact_replacement_required": True,
            "renewed_stage13_approval_granted_by_this_amendment": False,
            "stage13_execution_authorized_now": False,
        }
        or amendment.get("stage13_source_checkpoint_context_migration")
        != {
            "path": (
                "configs/phase_06_final_test/"
                "stage_13_source_checkpoint_context_migration_v1.json"
            ),
            "sha256": (_EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V1_SHA256),
            "memoization_authority_source_message_text": (
                "Stop the stage 13 run at the first possible checkpoint, "
                "memoizing everything so we don't lose anything"
            ),
            "memoization_authority_source_message_text_sha256": (
                "bdcaf0f7329cc5c12472e38e33a0abd77a767a2c96c041124f292b2a604bc678"
            ),
            "continuation_source_message_text": "Let's continue and finish stage 13",
            "continuation_source_message_text_sha256": (
                "035d476ed9117ac3fb07f055abf49085625118206f8e48db6c6a6c997c04935e"
            ),
            "parent_open_before_replacement_stage13_approval_permitted": False,
        }
        or tests.get("test_file")
        != "tests/unit/test_stage11_validation_empirical_feature_stream.py"
        or tests.get("must_cover")
        != [
            "positive_one_way_nullability_transport",
            "per_column_zero_null_receipt",
            "null_violation_rejection",
            "inverse_nullability_relaxation_rejection",
            "type_drift_rejection",
            "field_order_drift_rejection",
            "extension_frozen_schema_mismatch_rejection",
        ]
    ):
        raise Stage12FreezeError("post-unblind technical amendment contract differs")
    if sha256_bytes(str(authorization["source_message_text"]).encode("utf-8")) != (
        authorization["source_message_text_sha256"]
    ):
        raise Stage12FreezeError("post-unblind authorization message hash differs")
    if sha256_bytes(str(authorization["scope_request_text"]).encode("utf-8")) != (
        authorization["scope_request_text_sha256"]
    ):
        raise Stage12FreezeError("post-unblind scope-request message hash differs")
    hash_audit_incident_receipt = _authenticate_post_unblind_hash_audit_incident(
        hash_audit_incident,
        policy,
        project_root=project_root,
    )
    prior_authorities = _authenticate_prior_replaced_authorities(
        historical,
        project_root=project_root,
    )
    return amendment, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "amendment_id": str(amendment["amendment_id"]),
        "status": str(amendment["status"]),
        "authorization": dict(authorization),
        "source_transition": dict(transition),
        "schema_transport_contract": dict(transport),
        "non_economic_boundary": dict(boundary),
        "hash_audit_incident": hash_audit_incident_receipt,
        "replacement_authority": dict(replacement),
        "prior_replaced_authorities": prior_authorities,
    }


def _validate_post_unblind_scratch_topology_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the authorized non-mutating scratch-topology correction."""

    relative = (
        "configs/phase_06_final_test/"
        "stage_12_post_unblind_scratch_topology_amendment_v1.json"
    )
    expected_binding = {
        "path": relative,
        "sha256": _EXPECTED_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_SHA256,
    }
    _require_exact_mapping(
        _as_mapping(
            policy.get("post_unblind_scratch_topology_amendment"),
            "scratch-topology amendment binding is missing",
        ),
        expected_binding,
        label="scratch-topology amendment binding",
    )
    required = policy.get("required_authorities")
    amendment_authority = (
        [
            item
            for item in required
            if isinstance(item, Mapping)
            and item.get("role") == "post_unblind_scratch_topology_amendment"
        ]
        if isinstance(required, list)
        else []
    )
    blocker_authority = (
        [
            item
            for item in required
            if isinstance(item, Mapping)
            and item.get("role") == "stage13_parent_scratch_topology_blocker"
        ]
        if isinstance(required, list)
        else []
    )
    blocker_path = (
        "artifacts/phase_06_final_test/" "STAGE_13_PARENT_SCRATCH_TOPOLOGY_BLOCKER.json"
    )
    if amendment_authority != [
        {"role": "post_unblind_scratch_topology_amendment", **expected_binding}
    ] or blocker_authority != [
        {
            "role": "stage13_parent_scratch_topology_blocker",
            "path": blocker_path,
            "sha256": _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256,
        }
    ]:
        raise Stage12FreezeError("scratch-topology amendment authority differs")
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="scratch-topology amendment",
        )
        amendment, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("scratch-topology amendment path is unsafe") from exc
    authorization = _as_mapping(
        amendment.get("authorization"),
        "scratch-topology amendment authorization is missing",
    )
    authorization_receipt = _as_mapping(
        amendment.get("authorization_receipt"),
        "scratch-topology authorization receipt is missing",
    )
    historical = _as_mapping(
        amendment.get("historical_bindings"),
        "scratch-topology historical bindings are missing",
    )
    topology = _as_mapping(
        amendment.get("topology_contract"),
        "scratch-topology contract is missing",
    )
    source_scope = _as_mapping(
        amendment.get("source_scope"),
        "scratch-topology source scope is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "scratch-topology non-economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "scratch-topology replacement authority is missing",
    )
    tests = _as_mapping(
        amendment.get("required_tests"),
        "scratch-topology test contract is missing",
    )
    expected_descendants = [
        "left",
        "left/__stage13_exact_partition=27",
        "left/__stage13_exact_partition=79",
        "right",
        "right/__stage13_exact_partition=20",
        "right/__stage13_exact_partition=94",
    ]
    if (
        record.sha256 != _EXPECTED_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_SHA256
        or set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization",
            "authorization_receipt",
            "historical_bindings",
            "topology_contract",
            "source_scope",
            "non_economic_boundary",
            "replacement_authority",
            "required_tests",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-parent-scratch-topology-v1"
        or amendment.get("status")
        != "STAGE12_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED"
        or amendment.get("classification")
        != "non_economic_fail_closed_scratch_topology_contract_correction"
        or authorization
        != {
            "source": "current_codex_task_user_message",
            "authorization_form": "contextual_affirmative_reply_after_clarification",
            "scope_request_authored_by_user": False,
            "scope_request_text": (
                "I authorize the narrow, non-mutating Stage 12 scratch-topology "
                "amendment described in the blocker record: preserve the parent "
                "unchanged, authenticate the exact six empty directories, add "
                "topology-drift tests, refreeze and independently verify Stage 12. "
                "Ask me again before restarting Stage 13."
            ),
            "scope_request_text_sha256": (
                "21dcc932f427e7fac30d6b89034f92670e228af55486793a84d82abc1d46cf81"
            ),
            "source_message_text": "Okay lets start",
            "source_message_text_sha256": (
                "eadc45ff3c3a792931392d1491b0065e17804a1ef93eed54b82901d9c047c1d5"
            ),
            "authorized": True,
            "authorized_scope": (
                "only_the_non_mutating_exact_parent_scratch_topology_contract_"
                "correction_tests_and_replacement_stage12_refreeze_and_"
                "independent_verification"
            ),
            "parent_mutation_authorized": False,
            "destructive_cleanup_authorized": False,
            "renewed_stage13_approval_granted": False,
            "stage13_execution_authorized_now": False,
        }
        or authorization_receipt
        != {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZATION.json"
            ),
            "sha256": _EXPECTED_SCRATCH_TOPOLOGY_AUTHORIZATION_SHA256,
            "status": ("STAGE12_NON_MUTATING_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED"),
        }
        or historical
        != {
            "prior_policy_v2_path": (
                "configs/phase_06_final_test/stage_12_freeze_policy_v2.json"
            ),
            "prior_policy_v2_sha256": _EXPECTED_PRIOR_V2_REPLACEMENT_POLICY_SHA256,
            "prior_stage12_primary_pointer_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_PRIMARY_PRIOR_FREEZE_A0B24EE3.json"
            ),
            "prior_stage12_primary_pointer_sha256": (
                "04d1105329b09ea078f1abb4c7842ce6441b7c9f70d7869a11bf4954ced274fe"
            ),
            "prior_stage12_verified_pointer_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_VERIFIED_PRIOR_FREEZE_A0B24EE3.json"
            ),
            "prior_stage12_verified_pointer_sha256": (
                "850ba5805753406f9e7002e1cb71f8dfdea51c5d22606c80851baef8929384a6"
            ),
            "prior_stage12_freeze_id": "stage12-freeze-a0b24ee3eea641805a798dae",
            "prior_stage12_manifest_sha256": (
                "8fde217d8200355b1cd5dcfa3f2a6fcb3d57bb4591deda1f323dcd75e8023b16"
            ),
            "prior_stage12_verification_id": (
                "stage12-verify-2c0a3632f0ddfd77235aabfc"
            ),
            "prior_stage13_approval_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_APPROVAL_PRIOR_STAGE12_FREEZE_A0B24EE3.json"
            ),
            "prior_stage13_approval_sha256": (
                "3eece897f35e123659bc2fbf5721b42122d8be0394363d2fe31137c8d53548f2"
            ),
            "prior_stage13_approval_payload_sha256": (
                "00aa26a946504bf0eb449a6bb0c4ff9b50eaf80dbe565cee2777e019fbf01e55"
            ),
            "blocker_path": blocker_path,
            "blocker_sha256": (
                _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256
            ),
            "failed_stage13_context_sha256": (
                "7bc6e0a792c842c4763cf986d0b6acb8b6da857cfce51a7b9df3cc201b83b9b9"
            ),
            "failed_stage13_final_calendar_receipt_count": 72,
            "failed_stage13_economic_or_outcome_month_checkpoint_count": 0,
        }
        or topology.get("root_relative_path")
        != ("full_predictor_context/.duckdb_spill/" "exact-table-compare-1ejjjtbr")
        or topology.get("exact_descendant_relative_paths") != expected_descendants
        or topology.get("exact_descendant_directory_count") != 6
        or topology.get("root_and_descendant_directory_identity_count") != 7
        or any(
            topology.get(name) != 0
            for name in (
                "regular_file_count",
                "payload_bytes",
                "symlink_count",
                "special_node_count",
            )
        )
        or topology.get("missing_extra_renamed_or_type_drift_permitted") is not False
        or topology.get("stable_root_and_descendant_identity_before_after_required")
        is not True
        or topology.get("directory_identity_fields")
        != [
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ]
        or topology.get("parent_access_read_only") is not True
        or topology.get("parent_entry_deletion_move_rename_or_write_permitted")
        is not False
        or source_scope
        != {
            "economic_model_formula_registry_or_methodology_change_permitted": False,
            "data_rows_values_dates_types_keys_linkage_or_sort_change_permitted": False,
            "new_migration_authority_required": (
                "configs/phase_06_final_test/"
                "stage_13_source_checkpoint_context_migration_v2.json"
            ),
            "new_stage12_policy_required": (
                "configs/phase_06_final_test/stage_12_freeze_policy_v3.json"
            ),
            "runtime_validator_path": (
                "src/portfolio_bot/final_test/final_source_replay.py"
            ),
            "synthetic_test_path": (
                "tests/unit/test_stage13_source_checkpoint_context_migration.py"
            ),
        }
        or boundary
        != {
            "parent_entries_deleted_moved_renamed_or_mutated": False,
            "parent_payload_files_opened_by_stage12_amendment_work": 0,
            "outcome_or_economic_payload_files_opened_by_stage12_amendment_work": 0,
            "final_period_rows_read_by_stage12_amendment_work": 0,
            "economic_or_outcome_months_executed_by_stage12_amendment_work": 0,
            "final_forecasts_weights_trades_or_pnl_created_by_stage12_amendment_work": 0,
            "scientific_or_economic_result_informed_choice": False,
        }
        or replacement
        != {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_explicit_stage13_approval_bound_to_exact_replacement_required": True,
            "renewed_stage13_approval_granted_by_this_amendment": False,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_execution_authorized_now": False,
        }
        or tests.get("test_file")
        != "tests/unit/test_stage13_source_checkpoint_context_migration.py"
        or tests.get("must_cover")
        != [
            "exact_nested_payload_empty_topology_acceptance",
            "missing_renamed_or_extra_directory_rejection",
            "regular_file_symlink_and_special_node_rejection",
            "root_and_descendant_identity_drift_rejection",
            "before_after_parent_identity_equality",
            "receipt_reuse_reauthentication",
            "zero_parent_writes",
        ]
    ):
        raise Stage12FreezeError("scratch-topology amendment contract differs")
    if (
        sha256_bytes(str(authorization["source_message_text"]).encode("utf-8"))
        != (authorization["source_message_text_sha256"])
        or sha256_bytes(str(authorization["scope_request_text"]).encode("utf-8"))
        != authorization["scope_request_text_sha256"]
    ):
        raise Stage12FreezeError("scratch-topology authorization hash differs")

    try:
        authorization_receipt_path = confined_relative_path(
            project_root,
            str(authorization_receipt["path"]),
            file_required=True,
            label="scratch-topology authorization receipt",
        )
        authorization_document, authorization_record = load_json_object(
            authorization_receipt_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "scratch-topology authorization receipt is unsafe"
        ) from exc
    receipt_authorization = _as_mapping(
        authorization_document.get("authorization"),
        "scratch-topology receipt authorization is missing",
    )
    current_authority = _as_mapping(
        authorization_document.get("current_authority"),
        "scratch-topology receipt current authority is missing",
    )
    receipt_blocker = _as_mapping(
        authorization_document.get("blocker"),
        "scratch-topology receipt blocker is missing",
    )
    permitted_actions = _as_mapping(
        authorization_document.get("permitted_actions"),
        "scratch-topology receipt permitted actions are missing",
    )
    prohibited_actions = _as_mapping(
        authorization_document.get("prohibited_actions"),
        "scratch-topology receipt prohibited actions are missing",
    )
    handoff = _as_mapping(
        authorization_document.get("handoff"),
        "scratch-topology receipt handoff is missing",
    )
    shared_authorization_keys = (
        "source",
        "authorization_form",
        "scope_request_authored_by_user",
        "scope_request_text",
        "scope_request_text_sha256",
        "source_message_text",
        "source_message_text_sha256",
        "authorized",
        "authorized_scope",
    )
    if (
        authorization_record.sha256 != _EXPECTED_SCRATCH_TOPOLOGY_AUTHORIZATION_SHA256
        or set(authorization_document)
        != {
            "schema_version",
            "stage",
            "status",
            "authorization_id",
            "authorized_at_utc",
            "authorized_at_local",
            "authorization",
            "current_authority",
            "blocker",
            "permitted_actions",
            "prohibited_actions",
            "handoff",
        }
        or authorization_document.get("schema_version") != 1
        or authorization_document.get("stage") != 12
        or authorization_document.get("status")
        != "STAGE12_NON_MUTATING_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED"
        or authorization_document.get("authorization_id")
        != "stage12-scratch-topology-amendment-authorization-20260726-v1"
        or authorization_document.get("authorized_at_utc") != "2026-07-26T12:17:11Z"
        or authorization_document.get("authorized_at_local")
        != "2026-07-26T15:17:11+03:00"
        or set(receipt_authorization)
        != set(shared_authorization_keys) | {"replacement_stage13_approval_granted"}
        or any(
            receipt_authorization.get(key) != authorization.get(key)
            for key in shared_authorization_keys
        )
        or receipt_authorization.get("replacement_stage13_approval_granted")
        is not False
        or current_authority
        != {
            "stage12_freeze_id": "stage12-freeze-a0b24ee3eea641805a798dae",
            "stage12_manifest_sha256": (
                "8fde217d8200355b1cd5dcfa3f2a6fcb3d57bb4591deda1f323dcd75e8023b16"
            ),
            "stage12_primary_pointer_sha256": (
                "04d1105329b09ea078f1abb4c7842ce6441b7c9f70d7869a11bf4954ced274fe"
            ),
            "stage12_verification_id": "stage12-verify-2c0a3632f0ddfd77235aabfc",
            "stage12_verified_pointer_sha256": (
                "850ba5805753406f9e7002e1cb71f8dfdea51c5d22606c80851baef8929384a6"
            ),
            "stage12_binding_sha256": (
                "bb6462d608b17e651211984856a89c41e69871794ba442e2bf19c34c6a68c34f"
            ),
            "stage13_approval_sha256": (
                "3eece897f35e123659bc2fbf5721b42122d8be0394363d2fe31137c8d53548f2"
            ),
        }
        or receipt_blocker
        != {
            "path": blocker_path,
            "sha256": _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256,
            "classification": (
                "pre_existing_non_economic_contract_capture_and_synthetic_"
                "test_coverage_defect"
            ),
            "regular_file_count": 0,
            "payload_file_bytes": 0,
            "descendant_directory_count": 6,
        }
        or permitted_actions
        != {
            "exact_pin_existing_recursively_payload_empty_ordinary_directory_topology": True,
            "reject_missing_renamed_extra_file_symlink_or_special_entries": True,
            "compare_every_scratch_directory_identity_before_and_after_migration": True,
            "add_synthetic_adversarial_tests": True,
            "publish_replacement_stage12_freeze": True,
            "independently_verify_replacement_stage12_freeze": True,
        }
        or prohibited_actions
        != {
            "delete_move_rename_or_mutate_parent_scratch_entries": True,
            "open_or_parse_parent_payload_rows_during_amendment_work": True,
            "change_data_rows_values_dates_types_keys_linkage_formulas_models_or_economics": True,
            "execute_stage13_economic_or_outcome_months": True,
            "treat_this_authorization_as_replacement_stage13_approval": True,
        }
        or handoff
        != {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "current_stage13_approval_valid_for_replacement_identity": False,
            "renewed_stage13_approval_required_after_verification": True,
            "stop_and_ask_before_restarting_stage13": True,
        }
    ):
        raise Stage12FreezeError(
            "scratch-topology authorization receipt contract differs"
        )

    stable_bindings = {
        str(historical["prior_policy_v2_path"]): str(
            historical["prior_policy_v2_sha256"]
        ),
        str(historical["prior_stage12_primary_pointer_path"]): str(
            historical["prior_stage12_primary_pointer_sha256"]
        ),
        str(historical["prior_stage12_verified_pointer_path"]): str(
            historical["prior_stage12_verified_pointer_sha256"]
        ),
        str(historical["prior_stage13_approval_path"]): str(
            historical["prior_stage13_approval_sha256"]
        ),
        blocker_path: _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256,
    }
    for bound_relative, expected_sha in stable_bindings.items():
        try:
            bound_path = confined_relative_path(
                project_root,
                bound_relative,
                file_required=True,
                label="scratch-topology historical authority",
            )
            bound_record = stable_file(bound_path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "scratch-topology historical authority is unsafe"
            ) from exc
        if bound_record.sha256 != expected_sha:
            raise Stage12FreezeError("scratch-topology historical authority differs")
    return amendment, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "amendment_id": str(amendment["amendment_id"]),
        "status": str(amendment["status"]),
        "authorization": dict(authorization),
        "authorization_receipt": {
            "relative_path": str(authorization_receipt["path"]),
            "sha256": authorization_record.sha256,
            "size_bytes": authorization_record.size_bytes,
            "status": str(authorization_document["status"]),
            "authorization_id": str(authorization_document["authorization_id"]),
        },
        "topology_contract": dict(topology),
        "non_economic_boundary": dict(boundary),
        "replacement_authority": dict(replacement),
        "blocker": {
            "relative_path": blocker_path,
            "sha256": _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256,
            "status": (
                "STAGE13_FAILED_CLOSED_ON_PARENT_SCRATCH_TOPOLOGY_"
                "CONTRACT_PREMISE_MISMATCH"
            ),
        },
    }


def _validate_post_unblind_zero_row_merge_provenance_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the narrow zero-row fresh-materialization correction."""

    relative = (
        "configs/phase_06_final_test/"
        "stage_12_post_unblind_zero_row_merge_provenance_amendment_v1.json"
    )
    expected_binding = {
        "path": relative,
        "sha256": (_EXPECTED_POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_SHA256),
    }
    _require_exact_mapping(
        _as_mapping(
            policy.get("post_unblind_zero_row_merge_provenance_amendment"),
            "zero-row merge provenance amendment binding is missing",
        ),
        expected_binding,
        label="zero-row merge provenance amendment binding",
    )
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="zero-row merge provenance amendment",
        )
        amendment, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "zero-row merge provenance amendment path is unsafe"
        ) from exc

    authorization_receipt = _as_mapping(
        amendment.get("authorization_receipt"),
        "zero-row merge provenance authorization receipt is missing",
    )
    blocker_receipt = _as_mapping(
        amendment.get("blocker_receipt"),
        "zero-row merge provenance blocker receipt is missing",
    )
    historical = _as_mapping(
        amendment.get("historical_bindings"),
        "zero-row merge provenance historical bindings are missing",
    )
    scientific_scope = _as_mapping(
        amendment.get("frozen_scientific_scope"),
        "zero-row merge provenance scientific scope is missing",
    )
    correction = _as_mapping(
        amendment.get("provenance_contract"),
        "zero-row merge provenance contract is missing",
    )
    transition = _as_mapping(
        amendment.get("source_transition"),
        "zero-row merge provenance source transition is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "zero-row merge provenance non-economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "zero-row merge provenance replacement authority is missing",
    )
    tests = _as_mapping(
        amendment.get("required_tests"),
        "zero-row merge provenance test contract is missing",
    )
    inactive_features = [
        "business_segment_count",
        "geographic_segment_count",
        "business_segment_revenue_hhi",
        "business_segment_revenue_hhi_change_1y",
    ]
    expected_transition_members = [
        {
            "path": (
                "src/portfolio_bot/validation/" "validation_feature_input_adapter.py"
            ),
            "pre_amendment_sha256": _EXPECTED_ZERO_ROW_ADAPTER_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_ZERO_ROW_ADAPTER_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/final_source_replay.py",
            "pre_amendment_sha256": _EXPECTED_ZERO_ROW_REPLAY_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_ZERO_ROW_REPLAY_POST_SHA256,
        },
    ]
    authorization_path = (
        "artifacts/phase_06_final_test/"
        "STAGE_12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZATION.json"
    )
    blocker_path = (
        "artifacts/phase_06_final_test/" "STAGE_13_ZERO_ROW_COPY_GUARD_BLOCKER.json"
    )
    if (
        record.sha256
        != _EXPECTED_POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_SHA256
        or set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization_receipt",
            "blocker_receipt",
            "historical_bindings",
            "frozen_scientific_scope",
            "provenance_contract",
            "source_transition",
            "non_economic_boundary",
            "replacement_authority",
            "required_tests",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-zero-row-merge-provenance-v1"
        or amendment.get("status")
        != "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED"
        or amendment.get("classification")
        != "non_economic_deterministic_fresh_materialization_provenance_correction"
        or authorization_receipt
        != {
            "path": authorization_path,
            "sha256": _EXPECTED_ZERO_ROW_MERGE_PROVENANCE_AUTHORIZATION_SHA256,
            "status": "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED",
            "authorization_id": (
                "stage12-zero-row-merge-provenance-amendment-"
                "authorization-20260726-v1"
            ),
        }
        or blocker_receipt
        != {
            "path": blocker_path,
            "sha256": _EXPECTED_STAGE13_ZERO_ROW_COPY_GUARD_BLOCKER_SHA256,
            "status": "BLOCKED_PENDING_NARROW_TECHNICAL_AMENDMENT_AUTHORIZATION",
        }
        or historical
        != {
            "prior_policy_v3_path": (
                "configs/phase_06_final_test/stage_12_freeze_policy_v3.json"
            ),
            "prior_policy_v3_sha256": _EXPECTED_PRIOR_REPLACEMENT_POLICY_SHA256,
            "prior_file_inventory_path": (
                "artifacts/phase_06_final_test/stage_12_freezes/"
                "stage12-freeze-5fa8538a56b4ae01932f0705/file_inventory.json"
            ),
            "prior_file_inventory_sha256": _PRIOR_STAGE12_FILE_INVENTORY_SHA256,
            "prior_stage12_manifest_path": (
                "artifacts/phase_06_final_test/stage_12_freezes/"
                "stage12-freeze-5fa8538a56b4ae01932f0705/pre_final_manifest.json"
            ),
            "prior_stage12_manifest_sha256": (
                "b7c9bf2fe5ae670dec12a3a110e5a8b5e201a2ebafbed33b33c954b8c85489db"
            ),
            "prior_stage12_success_path": (
                "artifacts/phase_06_final_test/stage_12_freezes/"
                "stage12-freeze-5fa8538a56b4ae01932f0705/_SUCCESS.json"
            ),
            "prior_stage12_success_sha256": (
                "e81fa2ae6e4088676ec90676ff0f2f08969ebb86bb5f77ddbfb504932efa5630"
            ),
            "prior_stage12_verification_report_path": (
                "artifacts/phase_06_final_test/stage_12_verifications/"
                "stage12-verify-3eb98419d4491f711a4596b6/verification_report.json"
            ),
            "prior_stage12_verification_report_sha256": (
                "145118e189218ebfa32a3752b1398463f7add231b867102dd62698f9e1e8b85a"
            ),
            "prior_stage12_verification_success_path": (
                "artifacts/phase_06_final_test/stage_12_verifications/"
                "stage12-verify-3eb98419d4491f711a4596b6/_SUCCESS.json"
            ),
            "prior_stage12_verification_success_sha256": (
                "f281512c93b05336c26ac07d840b82c880240e665ec9c9a1502fc6e1f18a404c"
            ),
            "prior_stage13_approval_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_APPROVAL_CURRENT_STAGE12_FREEZE_5FA8538A.json"
            ),
            "prior_stage13_approval_sha256": (
                "b366335728dda2559b07acbcc4bb7ceb52bdcff64bb6d9993a934425cd1dec4e"
            ),
            "prior_stage13_approval_payload_sha256": (
                "16b733a76dfa30b1235e2a381c101e9aad55681e7d424da2d5e16f3cdd1fc089"
            ),
            "failed_stage13_launch_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_LAUNCH_CURRENT_STAGE12_5FA8538A.json"
            ),
            "failed_stage13_launch_sha256": (
                "2699fb26ad1b33f3733ff9009630b25e7941bc84587df0316ea9ebe9d1f6cdd0"
            ),
            "prior_stage12_freeze_id": "stage12-freeze-5fa8538a56b4ae01932f0705",
            "prior_stage12_primary_pointer_sha256": (
                "4c8bbec0ffc8c35319f661dacde9ddd292628a6c3a62598c6b863980801eb0c2"
            ),
            "prior_stage12_verification_id": (
                "stage12-verify-3eb98419d4491f711a4596b6"
            ),
            "prior_stage12_verified_pointer_sha256": (
                "5cda903338b0877222991084891afc79bfe277069e81408f1123dc85b93e05b5"
            ),
            "prior_stage12_binding_sha256": (
                "c1204044a2227ed23fcba9532651615850a60bd6a34993592fd432200a2d3c68"
            ),
            "failed_stage13_context_sha256": (
                "404b4c176a444adc44a8ebe279eee2e6d030df0a965eb73fc6f4f69c17dd8696"
            ),
            "failed_stage13_final_calendar_receipt_count": 72,
            "failed_stage13_economic_or_outcome_month_checkpoint_count": 0,
        }
        or scientific_scope
        != {
            "registered_feature_count": 163,
            "inactive_segment_feature_count": 4,
            "inactive_segment_features": inactive_features,
            "feature_registry_unchanged": True,
            "inactive_segment_support_unchanged": True,
            "rows_values_dates_types_keys_linkage_formulas_models_and_economics_unchanged": True,
        }
        or correction
        != {
            "scope": "only_zero_plus_zero_auxiliary_merges",
            "observed_collision_table": "compustat_segment_observation.parquet",
            "development_rows_required": 0,
            "extension_rows_required": 0,
            "result_rows_required": 0,
            "result_row_groups_required": 0,
            "merge_operator": "UNION ALL",
            "physical_provenance_metadata_key": (
                "portfolio_bot.zero_row_auxiliary_merge_provenance"
            ),
            "independent_output_inode_required": True,
            "output_sha256_must_differ_from_each_parent": True,
            "exact_field_schema_and_zero_rows_must_be_preserved": True,
            "copy_hardlink_symlink_inode_alias_and_nonordinary_rejection_retained": True,
            "nonzero_or_one_sided_merge_behavior_changed": False,
        }
        or transition
        != {
            "exact_changed_runtime_member_count": 2,
            "members": expected_transition_members,
        }
        or boundary
        != {
            "failed_context_or_sealed_parent_mutated": False,
            "parent_payload_files_opened_by_stage12_amendment_work": 0,
            "outcome_or_economic_payload_files_opened_by_stage12_amendment_work": 0,
            "final_period_rows_read_by_stage12_amendment_work": 0,
            "economic_or_outcome_months_executed_by_stage12_amendment_work": 0,
            "final_forecasts_weights_trades_or_pnl_created_by_stage12_amendment_work": 0,
            "scientific_or_economic_result_informed_choice": False,
        }
        or replacement
        != {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_explicit_stage13_approval_bound_to_exact_replacement_required": True,
            "renewed_stage13_approval_granted_by_this_amendment": False,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_execution_authorized_now": False,
        }
        or tests
        != {
            "test_files": [
                "tests/unit/test_stage11_validation_empirical_feature_stream.py",
                "tests/unit/test_stage13_source_checkpoint_context_migration.py",
            ],
            "must_cover": [
                "generic_zero_plus_zero_fresh_materialization_provenance",
                "exact_physical_provenance_authentication",
                "stale_or_tampered_provenance_rejection",
                "copy_hardlink_symlink_inode_alias_and_nonordinary_rejection",
                "nonzero_and_one_sided_merge_scope_exclusion",
                "frozen_163_feature_and_four_inactive_feature_preservation",
            ],
        }
    ):
        raise Stage12FreezeError("zero-row merge provenance amendment contract differs")

    stable_bindings = {
        str(historical[key]): str(historical[sha_key])
        for key, sha_key in (
            ("prior_policy_v3_path", "prior_policy_v3_sha256"),
            ("prior_file_inventory_path", "prior_file_inventory_sha256"),
            ("prior_stage12_manifest_path", "prior_stage12_manifest_sha256"),
            ("prior_stage12_success_path", "prior_stage12_success_sha256"),
            (
                "prior_stage12_verification_report_path",
                "prior_stage12_verification_report_sha256",
            ),
            (
                "prior_stage12_verification_success_path",
                "prior_stage12_verification_success_sha256",
            ),
            ("prior_stage13_approval_path", "prior_stage13_approval_sha256"),
            ("failed_stage13_launch_path", "failed_stage13_launch_sha256"),
        )
    }
    stable_bindings[str(authorization_receipt["path"])] = (
        _EXPECTED_ZERO_ROW_MERGE_PROVENANCE_AUTHORIZATION_SHA256
    )
    stable_bindings[str(blocker_receipt["path"])] = (
        _EXPECTED_STAGE13_ZERO_ROW_COPY_GUARD_BLOCKER_SHA256
    )
    loaded: dict[str, Mapping[str, Any]] = {}
    loaded_records: dict[str, Any] = {}
    for bound_relative, expected_sha in stable_bindings.items():
        try:
            bound_path = confined_relative_path(
                project_root,
                bound_relative,
                file_required=True,
                label="zero-row merge provenance authority",
            )
            value, bound_record = load_json_object(
                bound_path,
                trusted_root=project_root,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "zero-row merge provenance authority is unsafe"
            ) from exc
        if bound_record.sha256 != expected_sha:
            raise Stage12FreezeError("zero-row merge provenance authority differs")
        loaded[bound_relative] = value
        loaded_records[bound_relative] = bound_record

    authorization_document = loaded[authorization_path]
    blocker_document = loaded[blocker_path]
    authorization = _as_mapping(
        authorization_document.get("authorization"),
        "zero-row merge provenance authorization is missing",
    )
    authorized_scope = _as_mapping(
        authorization_document.get("frozen_scientific_scope"),
        "authorized zero-row scientific scope is missing",
    )
    prohibited = _as_mapping(
        authorization_document.get("prohibited_actions"),
        "zero-row provenance prohibited actions are missing",
    )
    handoff = _as_mapping(
        authorization_document.get("handoff"),
        "zero-row provenance handoff is missing",
    )
    if (
        authorization_document.get("schema_version") != 1
        or authorization_document.get("stage") != 12
        or authorization_document.get("status")
        != "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED"
        or authorization_document.get("authorization_id")
        != ("stage12-zero-row-merge-provenance-amendment-" "authorization-20260726-v1")
        or authorization.get("source_message_text") != "Yes"
        or authorization.get("source_message_text_sha256")
        != "85a39ab345d672ff8ca9b9c6876f3adcacf45ee7c1e2dbd2408fd338bd55e07e"
        or authorization.get("authorized") is not True
        or authorization.get("replacement_stage13_approval_granted") is not False
        or authorized_scope != scientific_scope
        or prohibited.get(
            "relax_or_remove_copy_hardlink_symlink_inode_alias_or_nonordinary_node_rejection"
        )
        is not True
        or prohibited.get("activate_or_remove_any_registered_feature") is not True
        or prohibited.get(
            "execute_stage13_under_a_replacement_stage12_identity_without_fresh_approval"
        )
        is not True
        or handoff
        != {
            "replacement_stage12_freeze_required": True,
            "independent_stage12_verification_required": True,
            "current_stage13_approval_valid_for_replacement_identity": False,
            "renewed_stage13_approval_required_after_verification": True,
            "stop_and_ask_before_restarting_stage13": True,
        }
        or blocker_document.get("schema_version") != 1
        or blocker_document.get("stage") != 13
        or blocker_document.get("status")
        != "BLOCKED_PENDING_NARROW_TECHNICAL_AMENDMENT_AUTHORIZATION"
        or blocker_document.get("root_cause")
        != (
            "The migration guard treats any child-parent byte hash equality as "
            "proof of copying. DuckDB independently materialized a schema-valid "
            "zero-row merge to a new inode, but deterministic zero-row Parquet "
            "encoding produced the same bytes as the parent zero-row development "
            "table. This is a false positive: no alias or copy occurred."
        )
        or blocker_document.get("exact_collision", {}).get("development_rows") != 0
        or blocker_document.get("exact_collision", {}).get("extension_rows") != 0
        or blocker_document.get("exact_collision", {}).get("result_rows") != 0
        or blocker_document.get("exact_collision", {}).get("same_inode") is not False
        or blocker_document.get("exact_collision", {}).get("hardlink_present")
        is not False
        or blocker_document.get("exact_collision", {}).get("symlink_present")
        is not False
        or blocker_document.get("fresh_context", {}).get("economic_checkpoint_count")
        != 0
        or blocker_document.get("fresh_context", {}).get("decision_barrier_present")
        is not False
    ):
        raise Stage12FreezeError("zero-row merge provenance authority semantics differ")

    return amendment, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "amendment_id": str(amendment["amendment_id"]),
        "status": str(amendment["status"]),
        "authorization_receipt": {
            "relative_path": authorization_path,
            "sha256": (loaded_records[authorization_path].sha256),
            "size_bytes": loaded_records[authorization_path].size_bytes,
            "status": str(authorization_document["status"]),
            "authorization_id": str(authorization_document["authorization_id"]),
        },
        "blocker_receipt": {
            "relative_path": blocker_path,
            "sha256": loaded_records[blocker_path].sha256,
            "size_bytes": loaded_records[blocker_path].size_bytes,
            "status": str(blocker_document["status"]),
        },
        "frozen_scientific_scope": dict(scientific_scope),
        "provenance_contract": dict(correction),
        "source_transition": dict(transition),
        "non_economic_boundary": dict(boundary),
        "replacement_authority": dict(replacement),
    }


def _validate_stage13_source_checkpoint_context_migration(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the closed, read-only Stage 13 prerequisite migration."""

    relative = (
        "configs/phase_06_final_test/"
        "stage_13_source_checkpoint_context_migration_v2.json"
    )
    expected_binding = {
        "path": relative,
        "sha256": _EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_SHA256,
    }
    _require_exact_mapping(
        _as_mapping(
            policy.get("stage13_source_checkpoint_context_migration"),
            "Stage 13 source-checkpoint migration binding is missing",
        ),
        expected_binding,
        label="Stage 13 source-checkpoint migration binding",
    )
    required = policy.get("required_authorities")
    authority = (
        [
            item
            for item in required
            if isinstance(item, Mapping)
            and item.get("role") == "stage13_source_checkpoint_context_migration"
        ]
        if isinstance(required, list)
        else []
    )
    if authority != [
        {
            "role": "stage13_source_checkpoint_context_migration",
            **expected_binding,
        }
    ]:
        raise Stage12FreezeError(
            "Stage 13 source-checkpoint migration authority differs"
        )
    post_unblind_amendment, _ = _validate_post_unblind_technical_amendment(
        policy,
        project_root=project_root,
    )
    scratch_amendment, _ = _validate_post_unblind_scratch_topology_amendment(
        policy,
        project_root=project_root,
    )
    v1_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_13_source_checkpoint_context_migration_v1.json"
        ),
        "sha256": _EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V1_SHA256,
    }
    if post_unblind_amendment.get("stage13_source_checkpoint_context_migration") != {
        **v1_binding,
        "memoization_authority_source_message_text": (
            "Stop the stage 13 run at the first possible checkpoint, "
            "memoizing everything so we don't lose anything"
        ),
        "memoization_authority_source_message_text_sha256": (
            "bdcaf0f7329cc5c12472e38e33a0abd77a767a2c96c041124f292b2a604bc678"
        ),
        "continuation_source_message_text": "Let's continue and finish stage 13",
        "continuation_source_message_text_sha256": (
            "035d476ed9117ac3fb07f055abf49085625118206f8e48db6c6a6c997c04935e"
        ),
        "parent_open_before_replacement_stage13_approval_permitted": False,
    }:
        raise Stage12FreezeError("post-unblind amendment migration binding differs")
    try:
        v1_path = confined_relative_path(
            project_root,
            v1_binding["path"],
            file_required=True,
            label="superseded Stage 13 source-checkpoint context migration",
        )
        v1_migration, v1_record = load_json_object(
            v1_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "superseded Stage 13 source-checkpoint migration path is unsafe"
        ) from exc
    if (
        v1_record.sha256
        != _EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V1_SHA256
        or set(v1_migration)
        != {
            "schema_version",
            "status",
            "migration_id",
            "parent_context",
            "authority_transition",
            "parent_feature_sources",
            "receipts",
            "excluded_parent_state",
            "safety",
        }
        or v1_migration.get("schema_version") != 1
        or v1_migration.get("status")
        != "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_AUTHORIZED"
        or v1_migration.get("migration_id")
        != "stage13-source-checkpoint-context-migration-v1"
    ):
        raise Stage12FreezeError(
            "superseded Stage 13 source-checkpoint migration identity differs"
        )
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="Stage 13 source-checkpoint context migration",
        )
        migration, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "Stage 13 source-checkpoint migration path is unsafe"
        ) from exc
    parent = _as_mapping(
        migration.get("parent_context"),
        "migration parent context is missing",
    )
    supersedes = _as_mapping(
        migration.get("supersedes"),
        "migration superseded authority is missing",
    )
    migration_amendment = _as_mapping(
        migration.get("scratch_topology_amendment"),
        "migration scratch-topology amendment is missing",
    )
    transition = _as_mapping(
        migration.get("authority_transition"),
        "migration authority transition is missing",
    )
    sources = _as_mapping(
        migration.get("parent_feature_sources"),
        "migration parent feature sources are missing",
    )
    subroots = _as_mapping(
        sources.get("subroots"),
        "migration source subroots are missing",
    )
    receipts = _as_mapping(
        migration.get("receipts"),
        "migration receipts are missing",
    )
    excluded_parent_state = _as_mapping(
        migration.get("excluded_parent_state"),
        "migration excluded parent state is missing",
    )
    safety = _as_mapping(
        migration.get("safety"),
        "migration safety contract is missing",
    )
    expected_subroots = {
        "development_auxiliary": {
            "file_count": 18,
            "total_size_bytes": 2383909536,
            "canonical_tree_sha256": (
                "031b805715ba43d39b8490222dafa54863d4cadccc5075a9dc7aed2637c43a17"
            ),
        },
        "universe_context": {
            "file_count": 4,
            "total_size_bytes": 284137691,
            "canonical_tree_sha256": (
                "dfac179e9347eecf8db58c43bfe4bd5ad35773979dcd9b6262c51d83e9a7ffd4"
            ),
        },
        "universe": {
            "file_count": 3,
            "total_size_bytes": 15225776,
            "canonical_tree_sha256": (
                "d55c7309cb1312877c6848b51d04048e8c99180adc948a2a7ace99e98dd2685c"
            ),
        },
        "auxiliary": {
            "file_count": 31,
            "total_size_bytes": 1840567690,
            "canonical_tree_sha256": (
                "a6d1ba7367553899943cacb3aef034c02f6f7f9d89c944f6bad56daa7ccb01a8"
            ),
        },
        "full_predictor_context": {
            "file_count": 8,
            "total_size_bytes": 2666793441,
            "canonical_tree_sha256": (
                "fab43e8ec6241da1fdcfdb7d3868a57e15664a255bb636d90c1ffcd4e2f52444"
            ),
        },
    }
    expected_receipts = {
        "universe_context": {
            "relative_path": "universe_context/context_receipt.json",
            "sha256": (
                "81ecb45774e7ec4675c25f0e934310402763cf4cee1627ca0c71251995527ea0"
            ),
        },
        "universe": {
            "relative_path": "universe/universe_auxiliary_receipt.json",
            "sha256": (
                "bc22455775a4a75f9aebeeb7091d51964edd51f1097ce7e8246b2d86dcadf0e1"
            ),
        },
        "auxiliary": {
            "relative_path": "auxiliary/universe_auxiliary_receipt.json",
            "sha256": (
                "0f326096930dc653dd246b0ac5193701dba73c364793dabd3b6f3f0f50d3a808"
            ),
        },
        "full_predictor_context": {
            "relative_path": (
                "full_predictor_context/full_predictor_context_receipt.json"
            ),
            "sha256": (
                "4fb499c8b89719c13ad169413d28844e5c19e1988665045acdb4f10e631a763c"
            ),
        },
    }
    if (
        record.sha256 != _EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_SHA256
        or set(migration)
        != {
            "schema_version",
            "status",
            "migration_id",
            "supersedes",
            "scratch_topology_amendment",
            "parent_context",
            "authority_transition",
            "parent_feature_sources",
            "receipts",
            "excluded_parent_state",
            "safety",
        }
        or migration.get("schema_version") != 1
        or migration.get("status")
        != "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V2_AUTHORIZED"
        or migration.get("migration_id")
        != "stage13-source-checkpoint-context-migration-v2"
        or supersedes
        != {
            **v1_binding,
            "migration_id": "stage13-source-checkpoint-context-migration-v1",
        }
        or migration_amendment
        != {
            "path": (
                "configs/phase_06_final_test/"
                "stage_12_post_unblind_scratch_topology_amendment_v1.json"
            ),
            "sha256": (_EXPECTED_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_SHA256),
            "amendment_id": "stage12-post-unblind-parent-scratch-topology-v1",
            "blocker_path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_PARENT_SCRATCH_TOPOLOGY_BLOCKER.json"
            ),
            "blocker_sha256": (
                _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256
            ),
        }
        or migration_amendment.get("amendment_id")
        != scratch_amendment.get("amendment_id")
        or parent
        != {
            "context_sha256": (
                "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5"
            ),
            "context_json_relative_path": "context.json",
            "context_json_sha256": (
                "04e42b584be2c70c49cfbc35f78358cdbb4af58544dec96869c46727567eb3d1"
            ),
            "raw_snapshot_id": "raw-v1-c1ab523852daf9e5",
            "raw_file_identities_sha256": (
                "cdea009462faaa6150ff24c76b57728dbb1a8d42cc496ba12632ae66b2e050f8"
            ),
            "stage12_binding_sha256": (
                "f9dbccbd448e5e36e52c83d43889549e7e12d4121d12ad049bc6a89962edc482"
            ),
            "approval_payload_sha256": (
                "256447acc775c383bb250c0ad0448a79696b51aee240e9417ec95311a8fc3a20"
            ),
        }
        or transition
        != {
            "parent_stage12_binding_sha256": (
                "f9dbccbd448e5e36e52c83d43889549e7e12d4121d12ad049bc6a89962edc482"
            ),
            "parent_approval_payload_sha256": (
                "256447acc775c383bb250c0ad0448a79696b51aee240e9417ec95311a8fc3a20"
            ),
            "replacement_stage12_binding_must_differ": True,
            "replacement_approval_payload_must_differ": True,
            "replacement_stage13_approval_required_before_parent_open": True,
            "raw_identity_must_match": True,
        }
        or set(sources)
        != {
            "relative_path",
            "allowed_prerequisite_directories",
            "forbidden_entries",
            "file_count",
            "total_size_bytes",
            "canonical_tree_sha256",
            "captured_tree_identity_method",
            "runtime_parent_before_after_digest_method",
            "subroots",
        }
        or sources.get("relative_path") != "formula_work/feature_sources"
        or sources.get("allowed_prerequisite_directories")
        != [
            "auxiliary",
            "development_auxiliary",
            "full_predictor_context",
            "universe",
            "universe_context",
        ]
        or sources.get("forbidden_entries")
        != [
            "feature_input_bundle",
            "final_predictors",
            "DECISION_SOURCES_COMMITTED.json",
            "months",
            "receipts",
        ]
        or sources.get("file_count") != 64
        or sources.get("total_size_bytes") != 7190634134
        or sources.get("canonical_tree_sha256")
        != "06483456c92d0302c124aba9f577fb17cc0d1330b6ab3dc5e08f7093ad89bac7"
        or sources.get("captured_tree_identity_method")
        != "sealed_pre_amendment_checkpoint_capture_v1"
        or sources.get("runtime_parent_before_after_digest_method")
        != ("sha256_of_canonical_newline_JSON_sorted_relative_path_sha256_size_records")
        or dict(subroots) != expected_subroots
        or dict(receipts) != expected_receipts
        or excluded_parent_state
        != {
            "only_parent_month_receipt_family": "final_calendar",
            "final_calendar_month_receipt_count": 72,
            "parent_final_calendar_receipts_reused": False,
            "new_context_final_calendar_recomputed": True,
            "final_predictor_month_receipt_count": 0,
            "economic_or_outcome_month_checkpoint_count": 0,
            "feature_input_bundle_present": False,
            "decision_source_barrier_present": False,
        }
        or safety
        != {
            "parent_access_read_only": True,
            "parent_tree_must_be_identical_before_after": True,
            "migration_created_hardlinks_copies_symlinks_forbidden": True,
            "feature_bundle_reuse_forbidden": True,
            "decision_outcome_economic_reuse_forbidden": True,
            "ignored_recursively_payload_empty_scratch_topologies": [
                {
                    "relative_path": (
                        "full_predictor_context/.duckdb_spill/"
                        "exact-table-compare-1ejjjtbr"
                    ),
                    "exact_descendant_relative_paths": [
                        "left",
                        "left/__stage13_exact_partition=27",
                        "left/__stage13_exact_partition=79",
                        "right",
                        "right/__stage13_exact_partition=20",
                        "right/__stage13_exact_partition=94",
                    ],
                }
            ],
            "ignored_scratch_topology_must_match_exactly": True,
            "ignored_scratch_regular_file_count_required": 0,
            "ignored_scratch_payload_bytes_required": 0,
            "ignored_scratch_symlink_count_required": 0,
            "ignored_scratch_special_node_count_required": 0,
            "ignored_scratch_directory_identities_before_after_must_match": True,
            "ignored_scratch_entries_must_not_be_removed_or_mutated": True,
            "new_context_allowed_writes": [
                "feature_input_bundle",
                "stage13_source_checkpoint_context_migration_receipt.json",
            ],
            "stage12_tests_synthetic_only": True,
            "required_synthetic_test_nodes": [
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_composes_fresh_bundle_without_parent_writes"
                ),
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_rejects_context_tree_and_scratch_drift"
                ),
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_rejects_exact_scratch_topology_drift"
                ),
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_rejects_scratch_identity_drift_during_composition"
                ),
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_reauthenticates_scratch_topology_on_receipt_reuse"
                ),
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_rejects_adapter_or_null_proof_tamper"
                ),
                (
                    "tests/unit/test_stage13_source_checkpoint_context_migration.py::"
                    "test_parent_checkpoint_migration_receipt_reuse_is_idempotent_and_allows_later_progress"
                ),
            ],
        }
        or sum(int(value["file_count"]) for value in subroots.values()) != 64
        or sum(int(value["total_size_bytes"]) for value in subroots.values())
        != 7190634134
    ):
        raise Stage12FreezeError(
            "Stage 13 source-checkpoint context migration contract differs"
        )
    return migration, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "migration_id": str(migration["migration_id"]),
        "status": str(migration["status"]),
        "supersedes": dict(supersedes),
        "superseded_authority_receipt": {
            "relative_path": str(v1_binding["path"]),
            "sha256": v1_record.sha256,
            "size_bytes": v1_record.size_bytes,
            "migration_id": str(v1_migration["migration_id"]),
            "status": str(v1_migration["status"]),
        },
        "scratch_topology_amendment": dict(migration_amendment),
        "parent_context": dict(parent),
        "authority_transition": dict(transition),
        "parent_feature_sources": dict(sources),
        "receipts": dict(receipts),
        "excluded_parent_state": dict(excluded_parent_state),
        "safety": dict(safety),
    }


def _validate_post_unblind_december_market_input_scope_partition_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the v5 scope repair without opening empirical payloads."""

    expected_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_december_market_input_scope_partition_"
            "amendment_v1.json"
        ),
        "sha256": _EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_SHA256,
    }
    expected_rebind_binding = {
        "path": (
            "configs/phase_06_final_test/" "stage_13_decision_checkpoint_rebind_v2.json"
        ),
        "sha256": _EXPECTED_STAGE13_DECISION_CHECKPOINT_REBIND_SHA256,
    }
    embedded_rebind_binding = {
        "path": (
            "configs/phase_06_final_test/" "stage_13_decision_checkpoint_rebind_v1.json"
        ),
        "sha256": _EXPECTED_STAGE13_DECISION_CHECKPOINT_REBIND_V1_SHA256,
    }
    metadata_exception_bindings = [
        {
            "role": "checkpoint_rebind_metadata_correction_authorization",
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_CHECKPOINT_REBIND_METADATA_CORRECTION_AMENDMENT_"
                "AUTHORIZATION.json"
            ),
            "sha256": _EXPECTED_CHECKPOINT_REBIND_METADATA_AUTHORIZATION_SHA256,
        },
        {
            "role": "stage13_checkpoint_rebind_metadata_blocker",
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_CHECKPOINT_REBIND_METADATA_BLOCKER_20260727.json"
            ),
            "sha256": _EXPECTED_CHECKPOINT_REBIND_METADATA_BLOCKER_SHA256,
        },
        {
            "role": "stage13_checkpoint_rebind_metadata_preflight",
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_CHECKPOINT_REBIND_METADATA_PREFLIGHT_20260727.json"
            ),
            "sha256": _EXPECTED_CHECKPOINT_REBIND_METADATA_PREFLIGHT_SHA256,
        },
    ]
    _require_exact_mapping(
        _as_mapping(
            policy.get("post_unblind_december_market_input_scope_partition_amendment"),
            "December market-input scope-partition amendment binding is missing",
        ),
        expected_binding,
        label="December market-input scope-partition amendment binding",
    )
    _require_exact_mapping(
        _as_mapping(
            policy.get("stage13_decision_checkpoint_rebind"),
            "decision-checkpoint rebind binding is missing",
        ),
        expected_rebind_binding,
        label="decision-checkpoint rebind binding",
    )
    authorities = policy.get("required_authorities")
    if not isinstance(authorities, list):
        raise Stage12FreezeError("Stage 12 authority list is missing")
    exact_authorities = [
        item
        for item in authorities
        if isinstance(item, Mapping)
        and item.get("role")
        in {
            "post_unblind_december_market_input_scope_partition_amendment",
            "stage13_december_market_input_blocker",
            "stage13_decision_checkpoint_rebind",
            "checkpoint_rebind_metadata_correction_authorization",
            "stage13_checkpoint_rebind_metadata_blocker",
            "stage13_checkpoint_rebind_metadata_preflight",
        }
    ]
    if exact_authorities != [
        {
            "role": "post_unblind_december_market_input_scope_partition_amendment",
            **expected_binding,
        },
        {
            "role": "stage13_december_market_input_blocker",
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_DECEMBER_MARKET_INPUT_BLOCKER_20260727.json"
            ),
            "sha256": _EXPECTED_STAGE13_DECEMBER_MARKET_INPUT_BLOCKER_SHA256,
        },
        {
            "role": "stage13_decision_checkpoint_rebind",
            **expected_rebind_binding,
        },
        *metadata_exception_bindings,
    ]:
        raise Stage12FreezeError("December repair direct authority set differs")

    def bound_json(
        binding: Mapping[str, Any], *, label: str
    ) -> tuple[Mapping[str, Any], StableFile]:
        try:
            path = confined_relative_path(
                project_root,
                binding.get("path"),
                file_required=True,
                label=label,
            )
            value, record = load_json_object(path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if record.sha256 != binding.get("sha256"):
            raise Stage12FreezeError(f"{label} SHA-256 differs")
        return value, record

    amendment, amendment_file = bound_json(
        expected_binding,
        label="December market-input scope-partition amendment",
    )
    if (
        amendment_file.sha256
        != _EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_SHA256
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("status")
        != "STAGE12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_AUTHORIZED"
        or amendment.get("amendment_id")
        != "stage12-post-unblind-december-market-input-scope-partition-v1"
    ):
        raise Stage12FreezeError("December repair amendment identity differs")

    authorization_binding = _as_mapping(
        amendment.get("authorization_receipt"),
        "December repair authorization binding is missing",
    )
    blocker_binding = _as_mapping(
        amendment.get("blocker_receipt"),
        "December blocker binding is missing",
    )
    prior = _as_mapping(
        amendment.get("prior_frozen_authority"),
        "December repair prior authority is missing",
    )
    scope = _as_mapping(
        amendment.get("decision_outcome_scope_partition"),
        "December scope partition is missing",
    )
    preflight = _as_mapping(
        amendment.get("preflight_and_completion_contract"),
        "December preflight contract is missing",
    )
    transition = _as_mapping(
        amendment.get("source_transition"),
        "December source transition is missing",
    )
    rebind_binding = _as_mapping(
        amendment.get("decision_checkpoint_rebind"),
        "decision-checkpoint rebind amendment binding is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "December non-economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "December replacement authority is missing",
    )
    tests = _as_mapping(amendment.get("tests"), "December repair tests are missing")
    _require_exact_mapping(
        authorization_binding,
        {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_"
                "AMENDMENT_AUTHORIZATION.json"
            ),
            "sha256": (
                _EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AUTHORIZATION_SHA256
            ),
            "authorization_id": (
                "stage12-december-market-input-scope-partition-amendment-"
                "authorization-20260727-v1"
            ),
            "source_message_exact": "yes, please fix these types of mistakes beforehand",
            "source_message_sha256": (
                "70c78b3f5dfe084851ae8072eaeb393f7c84c18c2380f920f4f80759b4013627"
            ),
            "replacement_stage13_approval_granted": False,
        },
        label="December repair authorization binding",
    )
    _require_exact_mapping(
        blocker_binding,
        {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_DECEMBER_MARKET_INPUT_BLOCKER_20260727.json"
            ),
            "sha256": _EXPECTED_STAGE13_DECEMBER_MARKET_INPUT_BLOCKER_SHA256,
            "classification": (
                "DETERMINISTIC_FROZEN_DECISION_OUTCOME_SCOPE_PARTITION_" "CONTRADICTION"
            ),
            "failed_table": "final_market_inputs",
            "missing_holding_month": "2025-12",
            "decision_source_receipts_present": 431,
            "decision_source_receipts_expected": 432,
        },
        label="December blocker binding",
    )
    _require_exact_mapping(
        scope,
        {
            "holding_month_count": 72,
            "first_holding_month": "2020-01",
            "last_holding_month": "2025-12",
            "last_signal_date": "2025-11-28",
            "exact_partial_entry_date": "2025-12-01",
            "last_exit_date": "2025-12-31",
            "decision_predictor_and_return_maximum_date": "2025-11-28",
            "entry_market_projection_exact_dates": ["2025-12-01"],
            "entry_market_projection_consumer_columns_only": True,
            "entry_market_projection_return_columns_requested": [],
            "entry_market_projection_labels_outcomes_accounting_or_performance_opened": False,
            "outcome_market_earliest_permitted_date": "2025-11-29",
            "outcome_sources_open_before_complete_decision_barrier": False,
            "calendar_dates_schemas_keys_features_models_and_economic_rules_changed": False,
        },
        label="December decision/outcome scope partition",
    )
    if set(preflight) != {
        "provider_preflight_runs_before_checkpoint_store_construction",
        "provider_preflight_requires_exact_72_month_calendar",
        "provider_preflight_authenticates_all_six_frozen_calendar_anchors",
        "provider_preflight_requires_non_circular_decision_sources",
        "provider_preflight_opens_outcome_sources",
        "family_commit_requires_built_month_set_equal_pending_month_set",
        "partial_or_missing_family_build_must_fail_before_first_fragment_commit",
        "invented_empty_or_synthetic_2025_12_receipt_forbidden",
    } or any(
        preflight.get(name) is not expected
        for name, expected in {
            "provider_preflight_runs_before_checkpoint_store_construction": True,
            "provider_preflight_requires_exact_72_month_calendar": True,
            "provider_preflight_authenticates_all_six_frozen_calendar_anchors": True,
            "provider_preflight_requires_non_circular_decision_sources": True,
            "provider_preflight_opens_outcome_sources": False,
            "family_commit_requires_built_month_set_equal_pending_month_set": True,
            "partial_or_missing_family_build_must_fail_before_first_fragment_commit": True,
            "invented_empty_or_synthetic_2025_12_receipt_forbidden": True,
        }.items()
    ):
        raise Stage12FreezeError("December preflight/completeness contract differs")

    expected_transition = {
        "exact_changed_existing_runtime_member_count": 2,
        "exact_added_runtime_member_count": 1,
        "members": [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": _EXPECTED_ZERO_ROW_REPLAY_POST_SHA256,
                "post_amendment_sha256": (
                    "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
                ),
            },
            {
                "path": ("src/portfolio_bot/validation/validation_market_extension.py"),
                "pre_amendment_sha256": (
                    "1bd9a4487351f1ac3337f2d3216f946ba553ee43a6006353ed1a019d7ad09446"
                ),
                "post_amendment_sha256": (
                    "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
                ),
            },
            {
                "path": ("src/portfolio_bot/final_test/source_checkpoint_rebind.py"),
                "pre_amendment_sha256": None,
                "post_amendment_sha256": (
                    "7216eb7b08b4cd0886e9ffff6497ed9a5b49c9013310c63e36a8f4e2d2d688bb"
                ),
            },
        ],
    }
    _require_exact_mapping(
        transition, expected_transition, label="December source transition"
    )
    for member in expected_transition["members"]:
        try:
            source = confined_relative_path(
                project_root,
                member["path"],
                file_required=True,
                label="December repaired runtime member",
            )
            source_record = stable_file(source, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "December repaired runtime member path is unsafe"
            ) from exc
        current_layer_sha256 = {
            "src/portfolio_bot/final_test/final_source_replay.py": (
                _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
            ),
            "src/portfolio_bot/validation/validation_market_extension.py": (
                _EXPECTED_ENTRY_MARKET_EXTENSION_POST_SHA256
            ),
            "src/portfolio_bot/final_test/source_checkpoint_rebind.py": (
                "7216eb7b08b4cd0886e9ffff6497ed9a5b49c9013310c63e36a8f4e2d2d688bb"
            ),
        }[str(member["path"])]
        if (
            not _is_consolidated_v12(policy)
            and source_record.sha256 != current_layer_sha256
        ):
            raise Stage12FreezeError(
                f"December repaired runtime member differs: {member['path']}"
            )

    expected_boundary = {
        "measurement_scope": (
            "guarded_stage12_publication_and_verification_subprocesses_only"
        ),
        "failed_context_or_sealed_parent_mutated": False,
        "guarded_parent_checkpoint_payload_files_opened": 0,
        "guarded_raw_payload_files_opened": 0,
        "guarded_final_period_payload_files_opened": 0,
        "guarded_final_period_rows_read": 0,
        "guarded_stage13_outcome_or_economic_payload_files_opened": 0,
        "guarded_economic_or_outcome_months_executed": 0,
        "guarded_final_forecasts_weights_trades_or_pnl_created": 0,
        "scientific_or_economic_result_informed_choice": False,
        "prior_unguarded_regression_access_disclosure": {
            "historical_access_detected": True,
            "operator_turn_history_minimum_known_whole_file_test_executions": 4,
            "persisted_execution_receipt_count": 0,
            "complete_low_level_open_event_count_recorded": False,
            "distinct_raw_payload_paths_opened": 1,
            "raw_payload_paths": [
                "/Users/agraemirolmez/Desktop/Thesis/raw/crsp_delisting_ciz.csv"
            ],
            "raw_physical_lines_read_per_execution": 2,
            "raw_record_period": "not_reauthenticated_unknown",
            "distinct_stage11_interim_payload_paths_opened": 3,
            "stage11_interim_payload_paths": [
                "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_validation/"
                "stage11-validation-data-v1/market_extension/typed/"
                "crsp_security_day_return_staging.parquet",
                "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_validation/"
                "stage11-validation-data-v1/market_extension/typed/"
                "crsp_security_day.parquet",
                "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_validation/"
                "stage11-validation-data-v1/market_extension/typed/"
                "crsp_security_month_return_staging.parquet",
            ],
            "stage11_logical_filter": {
                "permno": 16493,
                "start_date": "2016-11-01",
                "end_date": "2016-11-30",
            },
            "stage11_monthly_logical_rows_per_execution": 1,
            "other_logical_row_physical_row_group_and_byte_counts": "not_recorded",
            "stage13_parent_checkpoint_payload_paths_opened": 0,
            "stage13_final_context_opened": False,
            "stage13_outcome_sources_opened": False,
            "final_artifacts_created": 0,
            "payloads_mutated": False,
            "used_to_choose_december_rule_feature_model_or_economic_parameter": False,
            "corrective_action": (
                "whole_file_selector_replaced_by_exact_synthetic_nodes_under_the_"
                "stage12_access_guard"
            ),
        },
    }
    expected_replacement = {
        "replacement_stage12_freeze_required": True,
        "independent_stage12_verification_required": True,
        "prior_stage13_approval_valid_for_replacement_identity": False,
        "renewed_explicit_stage13_approval_bound_to_exact_replacement_required": True,
        "renewed_stage13_approval_granted_by_this_amendment": False,
        "stage13_parent_context_open_authorized_now": False,
        "stage13_execution_authorized_now": False,
        "stop_and_ask_before_restarting_stage13": True,
    }
    _require_exact_mapping(
        boundary, expected_boundary, label="December non-economic boundary"
    )
    _require_exact_mapping(
        replacement, expected_replacement, label="December replacement authority"
    )
    if tests.get("required_files") != [
        "tests/unit/test_final_source_builder.py",
        "tests/unit/test_stage11_validation_market_extension.py",
        "tests/unit/test_stage13_decision_checkpoint_rebind.py",
        "tests/unit/test_stage12_final_freeze.py",
    ] or any(
        tests.get(name) is not True
        for name in (
            "old_boundary_missing_december_rejected",
            "exact_entry_projection_and_72_month_completion_accepted",
            "provider_preflight_and_partial_build_fail_before_commit",
            "checkpoint_rebind_is_synthetic_only_during_stage12",
        )
    ):
        raise Stage12FreezeError("December repair test contract differs")

    authorization, authorization_file = bound_json(
        authorization_binding,
        label="December repair authorization receipt",
    )
    blocker, blocker_file = bound_json(
        blocker_binding, label="December market-input blocker"
    )
    if (
        authorization_file.sha256
        != _EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AUTHORIZATION_SHA256
        or authorization.get("status")
        != "STAGE12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_AUTHORIZED"
        or authorization.get("authorization_id")
        != authorization_binding["authorization_id"]
    ):
        raise Stage12FreezeError("December repair authorization identity differs")
    receipt_authorization = _as_mapping(
        authorization.get("authorization"),
        "December repair authorization semantics are missing",
    )
    preserved = _as_mapping(
        authorization.get("preserved_checkpoint_identity"),
        "December preserved checkpoint identity is missing",
    )
    handoff = _as_mapping(
        authorization.get("handoff"), "December authorization handoff is missing"
    )
    if (
        receipt_authorization.get("source_message_text")
        != authorization_binding["source_message_exact"]
        or receipt_authorization.get("source_message_text_sha256")
        != authorization_binding["source_message_sha256"]
        or receipt_authorization.get("authorized") is not True
        or receipt_authorization.get("replacement_stage13_approval_granted")
        is not False
        or preserved.get("failed_context_sha256")
        != "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e"
        or preserved.get("checkpoint_payload_count") != 431
        or preserved.get("checkpoint_receipt_count") != 431
        or preserved.get("decision_source_receipts_expected") != 432
        or preserved.get("missing_decision_source_receipts") != 1
        or preserved.get("decision_barrier_present") is not False
        or preserved.get("outcome_source_receipts_present") is not False
        or preserved.get("economic_checkpoint_count") != 0
        or handoff.get("current_stage13_approval_valid_for_replacement_identity")
        is not False
        or handoff.get("renewed_stage13_approval_required_after_verification")
        is not True
        or handoff.get("stop_and_ask_before_restarting_stage13") is not True
    ):
        raise Stage12FreezeError("December repair authorization semantics differ")

    blocker_failure = _as_mapping(
        blocker.get("failure"), "December blocker failure is missing"
    )
    blocker_static = _as_mapping(
        blocker.get("static_contradiction"),
        "December blocker contradiction is missing",
    )
    blocker_state = _as_mapping(
        blocker.get("durable_state"), "December blocker durable state is missing"
    )
    blocker_process = _as_mapping(
        blocker.get("process_control"), "December blocker process state is missing"
    )
    blocker_governance = _as_mapping(
        blocker.get("materiality_and_governance"),
        "December blocker governance is missing",
    )
    if (
        blocker_file.sha256 != _EXPECTED_STAGE13_DECEMBER_MARKET_INPUT_BLOCKER_SHA256
        or blocker.get("status")
        != "FAILED_CLOSED_BEFORE_DECISION_BARRIER_MISSING_FINAL_MARKET_INPUT_2025_12"
        or blocker.get("stage12_binding_sha256")
        != "4eba843d99bd2dd193a83383a66d34d430c0675da9e81e04fc0812a7e041eee8"
        or blocker.get("stage13_approval_sha256")
        != "2bc69b11d224f7915d0c63687daf1fa4f6212e97d725ed3602c6f8ef940277ea"
        or blocker.get("context_sha256")
        != "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e"
        or blocker_failure.get("classification") != blocker_binding["classification"]
        or blocker_failure.get("failed_table") != "final_market_inputs"
        or blocker_failure.get("missing_holding_month") != "2025-12"
        or blocker_failure.get("retry_under_same_binding_can_succeed") is not False
        or blocker_failure.get("resource_limit_related") is not False
        or blocker_static.get("frozen_2025_12_signal_date") != "2025-11-28"
        or blocker_static.get("frozen_2025_12_entry_date") != "2025-12-01"
        or blocker_static.get("frozen_2025_12_exit_date") != "2025-12-31"
        or blocker_static.get("raw_data_missing") is not False
        or blocker_static.get("outcome_market_open_before_decision_barrier_permitted")
        is not False
        or blocker_state.get("decision_source_receipts") != 431
        or blocker_state.get("expected_decision_source_receipts") != 432
        or blocker_state.get("decision_barrier_present") is not False
        or blocker_state.get("outcome_source_receipts_present") is not False
        or blocker_state.get("economic_checkpoint_count") != 0
        or blocker_process.get("stage13_process_running") is not False
        or blocker_governance.get("negligible_error_policy_applicable") is not False
        or blocker_governance.get("fresh_binding_specific_stage13_approval_required")
        is not True
    ):
        raise Stage12FreezeError("December blocker semantics differ")

    expected_prior = {
        "policy": {
            "path": "configs/phase_06_final_test/stage_12_freeze_policy_v4.json",
            "sha256": _EXPECTED_PRIOR_V4_REPLACEMENT_POLICY_SHA256,
        },
        "file_inventory": {
            "path": _PRIOR_V4_STAGE12_FILE_INVENTORY_RELATIVE_PATH,
            "sha256": _PRIOR_V4_STAGE12_FILE_INVENTORY_SHA256,
            "file_count": 595,
            "tree_sha256": (
                "f56241945d55fce817dd2ee25104c1b8714dbb778a11005a03a7e92842a862e4"
            ),
        },
        "primary_pointer_archive": {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_PRIMARY_PRIOR_FREEZE_8641D950.json"
            ),
            "sha256": (
                "72f1f01a00ea2789db5131172f1a8609da4295a00b27ee919b09528fca60debf"
            ),
            "freeze_id": "stage12-freeze-8641d95070c80372cfa2ad02",
        },
        "verified_pointer_archive": {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_VERIFIED_PRIOR_FREEZE_8641D950.json"
            ),
            "sha256": (
                "a333c7026c014a4a030a9a8f3548fb0fc2516f5786f4ec1023a9755a93db2c74"
            ),
            "verification_id": "stage12-verify-63585b576f3b60749d35956b",
        },
        "stage13_approval_archive": {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_APPROVAL_PRIOR_STAGE12_FREEZE_8641D950.json"
            ),
            "sha256": (
                "2bc69b11d224f7915d0c63687daf1fa4f6212e97d725ed3602c6f8ef940277ea"
            ),
            "payload_sha256": (
                "7948764f27166ebbc36ccacf9b9084eb5697b1e52ae0820c758e8dc8472311b8"
            ),
            "stage12_binding_sha256": (
                "4eba843d99bd2dd193a83383a66d34d430c0675da9e81e04fc0812a7e041eee8"
            ),
        },
        "failed_context": {
            "context_sha256": (
                "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e"
            ),
            "context_document_sha256": (
                "f7cf334957a8eb7c29d4b80250a4763cb1a42385977b1d9396d79ffee72982a7"
            ),
        },
    }
    _require_exact_mapping(
        prior, expected_prior, label="December prior frozen authority"
    )
    prior_documents: dict[str, Mapping[str, Any]] = {}
    prior_receipts: dict[str, dict[str, Any]] = {}
    for name in (
        "policy",
        "file_inventory",
        "primary_pointer_archive",
        "verified_pointer_archive",
        "stage13_approval_archive",
    ):
        document, record = bound_json(
            _as_mapping(prior[name], f"December prior {name} binding is missing"),
            label=f"December prior {name}",
        )
        prior_documents[name] = document
        prior_receipts[name] = {
            "relative_path": str(prior[name]["path"]),
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }
    prior_inventory = prior_documents["file_inventory"]
    prior_primary = prior_documents["primary_pointer_archive"]
    prior_verified = prior_documents["verified_pointer_archive"]
    prior_approval = prior_documents["stage13_approval_archive"]
    approval_payload = dict(prior_approval)
    approval_payload_sha = approval_payload.pop("payload_sha256", None)
    if (
        prior_inventory.get("file_count") != 595
        or prior_inventory.get("tree_sha256")
        != expected_prior["file_inventory"]["tree_sha256"]
        or prior_primary.get("freeze_id")
        != expected_prior["primary_pointer_archive"]["freeze_id"]
        or prior_primary.get("status")
        != "STAGE12_POST_UNBLIND_REPLACEMENT_FREEZE_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
        or prior_verified.get("verification_id")
        != expected_prior["verified_pointer_archive"]["verification_id"]
        or prior_verified.get("primary_pointer_sha256")
        != expected_prior["primary_pointer_archive"]["sha256"]
        or prior_verified.get("difference_count") != 0
        or prior_approval.get("approved") is not True
        or prior_approval.get("stage12_binding_sha256")
        != expected_prior["stage13_approval_archive"]["stage12_binding_sha256"]
        or approval_payload_sha
        != expected_prior["stage13_approval_archive"]["payload_sha256"]
        or sha256_bytes(canonical_json_bytes(approval_payload)) != approval_payload_sha
    ):
        raise Stage12FreezeError("December archived authority semantics differ")

    _require_exact_mapping(
        rebind_binding,
        {
            **embedded_rebind_binding,
            "rebind_id": "stage13-decision-checkpoint-v4-to-v5-rebind-v1",
            "checkpoint_count": 431,
            "receipt_count": 431,
            "total_rows": 1522126,
            "total_payload_bytes": 505136766,
            "fresh_receipts_change_only_context_sha256": True,
            "parent_access_read_only": True,
            "hardlinks_symlinks_and_aliases_forbidden": True,
            "post_completion_state_verify_only": True,
            "completion_receipt_published_last": True,
            "parent_decision_market_cache_reuse_mode": (
                "authenticated_read_only_direct_parent_paths"
            ),
            "parent_decision_market_cache_typed_receipt_sha256": (
                "1b287e50a9e7d2eb2ed7bfa2c4e84bf8dc52185899d7fda6bed6949589fc248e"
            ),
            "parent_decision_market_cache_return_receipt_sha256": (
                "3ef1a4c8a71d3f2ec8a2980da66fb9fe21ec23c845b71680f9ad7a7ddb90f619"
            ),
            "parent_decision_market_cache_fallback_rebuild_on_authentication_failure": False,
            "stage12_tests_synthetic_only": True,
        },
        label="December decision-checkpoint rebind binding",
    )
    rebind, rebind_file = bound_json(
        expected_rebind_binding, label="Stage 13 decision-checkpoint rebind"
    )
    rebind_parent = _as_mapping(
        rebind.get("parent_context"), "rebind parent context is missing"
    )
    rebind_aggregates = _as_mapping(
        rebind.get("aggregate_pins"), "rebind aggregate pins are missing"
    )
    rebind_excluded = _as_mapping(
        rebind.get("excluded_parent_state"), "rebind excluded state is missing"
    )
    rebind_safety = _as_mapping(
        rebind.get("safety"), "rebind safety contract is missing"
    )
    rebind_cache = _as_mapping(
        rebind.get("parent_decision_market_cache"),
        "rebind parent decision-market cache contract is missing",
    )
    rebind_cache_scope = _as_mapping(
        rebind_cache.get("scope"), "rebind parent cache scope is missing"
    )
    rebind_cache_safety = _as_mapping(
        rebind_cache.get("safety"), "rebind parent cache safety is missing"
    )
    _require_exact_mapping(
        rebind_cache,
        {
            "reuse_mode": "authenticated_read_only_direct_parent_paths",
            "relative_path": "formula_work/decision_market",
            "typed_relative_path": "typed",
            "typed_receipt_sha256": (
                "1b287e50a9e7d2eb2ed7bfa2c4e84bf8dc52185899d7fda6bed6949589fc248e"
            ),
            "returns_relative_path": "returns",
            "return_receipt_sha256": (
                "c01951d75930f6e46387781a78a452afb6a1c267081a3d06cacf204c99e3e7e1"
            ),
            "legacy_market_extension_source_sha256": (
                "1bd9a4487351f1ac3337f2d3216f946ba553ee43a6006353ed1a019d7ad09446"
            ),
            "reconciliation_source_sha256": (
                "18e8209005ecb4b2672bb4e62567e61a3920abe4a346104c2abe2163690a1872"
            ),
            "return_adapter_source_sha256": (
                "d5cc2046f607782625d3509a3f8acb3d8d63a78fdbfe8d43f3f60b25b63f1603"
            ),
            "scope": {
                "extension_start": "2019-11-01",
                "last_signal": "2025-11-28",
                "last_exit": "2025-12-31",
                "first_context_signal": "2019-10-31",
            },
            "safety": {
                "selected_subtrees_read_only": True,
                "selected_subtrees_reauthenticated_before_reuse": True,
                "selected_subtrees_identical_before_after": True,
                "unselected_siblings_not_opened": True,
                "hardlinks_symlinks_special_nodes_forbidden": True,
                "typed_and_return_builder_semantics_unchanged": True,
                "entry_projection_change_semantically_disjoint": True,
                "replacement_cache_copy_authorized": False,
                "fallback_rebuild_on_authentication_failure": False,
                "interrupted_siblings_may_not_be_opened_or_quarantined": True,
            },
        },
        label="rebind parent decision-market cache contract",
    )
    if (
        rebind_file.sha256 != _EXPECTED_STAGE13_DECISION_CHECKPOINT_REBIND_SHA256
        or rebind.get("authority_revision") != 2
        or rebind.get("authority_path") != expected_rebind_binding["path"]
        or rebind.get("supersedes")
        != {**embedded_rebind_binding, "authority_revision": 1}
        or rebind.get("status")
        != "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_AUTHORIZED"
        or rebind.get("rebind_id") != rebind_binding["rebind_id"]
        or rebind_parent.get("context_sha256")
        != expected_prior["failed_context"]["context_sha256"]
        or rebind_parent.get("context_json_sha256")
        != expected_prior["failed_context"]["context_document_sha256"]
        or rebind_parent.get("stage12_binding_sha256")
        != expected_prior["stage13_approval_archive"]["stage12_binding_sha256"]
        or rebind_parent.get("approval_payload_sha256")
        != expected_prior["stage13_approval_archive"]["payload_sha256"]
        or rebind_aggregates.get("checkpoint_count") != 431
        or rebind_aggregates.get("receipt_count") != 431
        or rebind_aggregates.get("total_rows") != 1522126
        or rebind_aggregates.get("total_payload_bytes") != 505136766
        or rebind_aggregates.get("logical_content_set_sha256")
        != "566500196d4c1744ec5ab1f83b029f3a1fc22015d5b62220b38cec1bd5fe4d26"
        or _as_mapping(
            rebind.get("source_families"), "rebind source families are missing"
        )
        .get("final_risk_cost_month_summary", {})
        .get("arrow_schema_sha256")
        != "9502c043b3b8d26c490a25757b684e6f4e4a9b7e1430df8ba0c68100cd0adc13"
        or rebind_aggregates.get("aggregate_method")
        != "sha256_sorted_relative_path_nul_binary_digest_v1"
        or rebind_excluded.get("final_market_inputs_2025_12_present") is not False
        or rebind_excluded.get("decision_source_barrier_present") is not False
        or rebind_excluded.get("outcome_source_checkpoint_count") != 0
        or rebind_excluded.get("economic_checkpoint_count") != 0
        or rebind_safety.get("parent_access_read_only") is not True
        or rebind_safety.get("payload_copy_bytes_exact") is not True
        or rebind_safety.get("hardlinks_symlinks_aliases_forbidden") is not True
        or rebind_safety.get("post_completion_state_verify_only") is not True
        or rebind_safety.get("completion_receipt_published_last") is not True
        or rebind_safety.get("stage12_tests_synthetic_only") is not True
        or rebind_safety.get(
            "production_parent_payloads_may_not_be_opened_during_stage12_validation"
        )
        is not True
    ):
        raise Stage12FreezeError("decision-checkpoint rebind semantics differ")

    metadata_exception_receipts: dict[str, dict[str, Any]] = {}
    expected_statuses = {
        "checkpoint_rebind_metadata_correction_authorization": (
            "STAGE12_CHECKPOINT_REBIND_METADATA_CORRECTION_AMENDMENT_AUTHORIZED"
        ),
        "stage13_checkpoint_rebind_metadata_blocker": (
            "STAGE13_FAILED_CLOSED_ON_CHECKPOINT_REBIND_METADATA_AUTHORITY_MISMATCH"
        ),
        "stage13_checkpoint_rebind_metadata_preflight": (
            "STAGE13_CHECKPOINT_REBIND_METADATA_PREFLIGHT_COMPLETE"
        ),
    }
    for binding in metadata_exception_bindings:
        role = str(binding["role"])
        document, record = bound_json(
            binding, label=f"checkpoint-rebind metadata exception {role}"
        )
        if (
            document.get("schema_version") != 1
            or document.get("status") != expected_statuses[role]
        ):
            raise Stage12FreezeError(
                "checkpoint-rebind metadata exception identity differs"
            )
        metadata_exception_receipts[role] = {
            "relative_path": binding["path"],
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
            "status": document["status"],
        }

    return amendment, {
        "relative_path": expected_binding["path"],
        "sha256": amendment_file.sha256,
        "size_bytes": amendment_file.size_bytes,
        "status": amendment["status"],
        "amendment_id": amendment["amendment_id"],
        "authorization_receipt": {
            "relative_path": authorization_binding["path"],
            "sha256": authorization_file.sha256,
            "size_bytes": authorization_file.size_bytes,
            "status": authorization["status"],
            "authorization_id": authorization["authorization_id"],
        },
        "blocker_receipt": {
            "relative_path": blocker_binding["path"],
            "sha256": blocker_file.sha256,
            "size_bytes": blocker_file.size_bytes,
            "status": blocker["status"],
            "classification": blocker_failure["classification"],
        },
        "prior_authority_receipts": prior_receipts,
        "decision_outcome_scope_partition": dict(scope),
        "preflight_and_completion_contract": dict(preflight),
        "source_transition": dict(transition),
        "decision_checkpoint_rebind": {
            "relative_path": expected_rebind_binding["path"],
            "sha256": rebind_file.sha256,
            "size_bytes": rebind_file.size_bytes,
            "status": rebind["status"],
            "rebind_id": rebind["rebind_id"],
            "authority_revision": rebind["authority_revision"],
            "supersedes": dict(rebind["supersedes"]),
            "aggregate_pins": dict(rebind_aggregates),
            "excluded_parent_state": dict(rebind_excluded),
            "safety": dict(rebind_safety),
            "parent_decision_market_cache": {
                **dict(rebind_cache),
                "scope": dict(rebind_cache_scope),
                "safety": dict(rebind_cache_safety),
            },
        },
        "checkpoint_rebind_metadata_exception": metadata_exception_receipts,
        "non_economic_boundary": dict(boundary),
        "replacement_authority": dict(replacement),
    }


def _validate_post_unblind_entry_market_canonicalization_scope_alignment_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the v6 entry-market alignment without empirical reads."""

    expected_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_entry_market_canonicalization_scope_"
            "alignment_amendment_v1.json"
        ),
        "sha256": _EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AMENDMENT_SHA256,
    }
    _require_exact_mapping(
        _as_mapping(
            policy.get(
                "post_unblind_entry_market_canonicalization_scope_alignment_amendment"
            ),
            "entry-market alignment amendment binding is missing",
        ),
        expected_binding,
        label="entry-market alignment amendment binding",
    )
    authorities = policy.get("required_authorities")
    if not isinstance(authorities, list):
        raise Stage12FreezeError("Stage 12 authority list is missing")
    direct_roles = {
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment",
        "stage13_entry_market_projection_contract_blocker",
        "stage13_entry_market_share_coverage_preflight",
    }
    direct = [
        item
        for item in authorities
        if isinstance(item, Mapping) and item.get("role") in direct_roles
    ]
    expected_direct = [
        {
            "role": (
                "post_unblind_entry_market_canonicalization_scope_alignment_"
                "amendment"
            ),
            **expected_binding,
        },
        {
            "role": "stage13_entry_market_projection_contract_blocker",
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_ENTRY_MARKET_PROJECTION_CONTRACT_BLOCKER_20260727.json"
            ),
            "sha256": _EXPECTED_STAGE13_ENTRY_MARKET_PROJECTION_BLOCKER_SHA256,
        },
        {
            "role": "stage13_entry_market_share_coverage_preflight",
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_ENTRY_MARKET_SHARE_COVERAGE_PREFLIGHT_20260727.json"
            ),
            "sha256": _EXPECTED_STAGE13_ENTRY_MARKET_SHARE_PREFLIGHT_SHA256,
        },
    ]
    if direct != expected_direct:
        raise Stage12FreezeError("entry-market alignment direct authority set differs")

    def bound_json(
        binding: Mapping[str, Any], *, label: str
    ) -> tuple[Mapping[str, Any], StableFile]:
        try:
            path = confined_relative_path(
                project_root,
                binding.get("path"),
                file_required=True,
                label=label,
            )
            value, record = load_json_object(path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if record.sha256 != binding.get("sha256"):
            raise Stage12FreezeError(f"{label} SHA-256 differs")
        return value, record

    amendment, amendment_file = bound_json(
        expected_binding,
        label="entry-market alignment amendment",
    )
    expected_root_keys = {
        "schema_version",
        "stage",
        "status",
        "amendment_id",
        "purpose",
        "authorization_receipt",
        "incident_authorities",
        "prior_frozen_authority",
        "entry_market_projection_contract",
        "exact_duplicate_canonicalization_contract",
        "closed_tombstone_share_coverage_contract",
        "receipt_contract",
        "source_transition",
        "non_economic_boundary",
        "replacement_authority",
        "tests",
    }
    if (
        amendment_file.sha256 != _EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AMENDMENT_SHA256
        or set(amendment) != expected_root_keys
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("status")
        != "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT_AUTHORIZED"
        or amendment.get("amendment_id")
        != "stage12-post-unblind-entry-market-canonicalization-scope-alignment-v1"
    ):
        raise Stage12FreezeError("entry-market alignment amendment identity differs")

    authorization_binding = _as_mapping(
        amendment.get("authorization_receipt"),
        "entry-market alignment authorization binding is missing",
    )
    incident = _as_mapping(
        amendment.get("incident_authorities"),
        "entry-market incident authority set is missing",
    )
    prior = _as_mapping(
        amendment.get("prior_frozen_authority"),
        "entry-market prior authority is missing",
    )
    projection = _as_mapping(
        amendment.get("entry_market_projection_contract"),
        "entry-market projection contract is missing",
    )
    canonicalization = _as_mapping(
        amendment.get("exact_duplicate_canonicalization_contract"),
        "entry-market canonicalization contract is missing",
    )
    coverage = _as_mapping(
        amendment.get("closed_tombstone_share_coverage_contract"),
        "entry-market share-coverage contract is missing",
    )
    receipt_contract = _as_mapping(
        amendment.get("receipt_contract"),
        "entry-market receipt contract is missing",
    )
    transition = _as_mapping(
        amendment.get("source_transition"),
        "entry-market source transition is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "entry-market non-economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "entry-market replacement authority is missing",
    )
    tests = _as_mapping(
        amendment.get("tests"),
        "entry-market test contract is missing",
    )

    _require_exact_mapping(
        authorization_binding,
        {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_"
                "AMENDMENT_AUTHORIZATION.json"
            ),
            "sha256": _EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AUTHORIZATION_SHA256,
            "size_bytes": 7513,
            "authorization_id": (
                "stage12-entry-market-canonicalization-scope-alignment-"
                "authorization-20260727-v1"
            ),
            "authorization_form": (
                "contextual_standing_approval_after_exact_scope_and_future_"
                "reapproval_disclosure"
            ),
            "source_message_id": (
                "thread-user-stage12-v6-standing-approval-20260727-01"
            ),
            "source_message_exact": "dont ask me these ill always approve",
            "source_message_utf8_size_bytes": 36,
            "source_message_has_terminal_newline": False,
            "source_message_sha256": (
                "c1ead3d6041ffbfad689976c6609a70754fbff4c04c05bad701ec57591bde91e"
            ),
            "current_v6_amendment_implementation_authorized_now": True,
            "standing_stage13_approval_granted": True,
            "fresh_binding_specific_stage13_receipt_issued_now": False,
            "stage13_execution_authorized_now": False,
            "additional_user_prompt_after_v6_verification_required": False,
        },
        label="entry-market alignment authorization binding",
    )
    expected_incident = {
        "exact_duplicate_blocker": {
            "path": (
                "artifacts/phase_06_final_test/"
                "STAGE_13_ENTRY_MARKET_EXACT_DUPLICATE_BLOCKER_20260727.json"
            ),
            "sha256": (
                "813f5f38a802eb0b926cdaca565016a27a40a4ac96b7e8cb0a89fc1f08bb3f46"
            ),
        },
        "combined_projection_blocker": {
            "path": expected_direct[1]["path"],
            "sha256": _EXPECTED_STAGE13_ENTRY_MARKET_PROJECTION_BLOCKER_SHA256,
            "status": (
                "BLOCKED_PENDING_NARROW_ENTRY_MARKET_CANONICALIZATION_AND_"
                "TOMBSTONE_COVERAGE_AMENDMENT_AUTHORIZATION"
            ),
        },
        "share_coverage_preflight": {
            "path": expected_direct[2]["path"],
            "sha256": _EXPECTED_STAGE13_ENTRY_MARKET_SHARE_PREFLIGHT_SHA256,
            "status": (
                "ENTRY_MARKET_SHARE_COVERAGE_PREFLIGHT_COMPLETE_NO_THIRD_"
                "PROJECTION_BLOCKER"
            ),
        },
    }
    _require_exact_mapping(
        incident,
        expected_incident,
        label="entry-market incident authority set",
    )

    authorization, authorization_file = bound_json(
        authorization_binding,
        label="entry-market alignment authorization receipt",
    )
    exact_blocker, exact_blocker_file = bound_json(
        _as_mapping(
            incident["exact_duplicate_blocker"],
            "exact-duplicate blocker binding is missing",
        ),
        label="entry-market exact-duplicate blocker",
    )
    blocker, blocker_file = bound_json(
        _as_mapping(
            incident["combined_projection_blocker"],
            "combined entry-market blocker binding is missing",
        ),
        label="combined entry-market projection blocker",
    )
    preflight, preflight_file = bound_json(
        _as_mapping(
            incident["share_coverage_preflight"],
            "entry-market preflight binding is missing",
        ),
        label="entry-market share-coverage preflight",
    )
    auth_semantics = _as_mapping(
        authorization.get("authorization"),
        "entry-market authorization semantics are missing",
    )
    standing = _as_mapping(
        authorization.get("standing_stage13_approval"),
        "standing Stage 13 approval semantics are missing",
    )
    auth_limits = _as_mapping(
        authorization.get("authorization_limits"),
        "entry-market authorization limits are missing",
    )
    authorized_runtime = _as_mapping(
        authorization.get("authorized_runtime_scope"),
        "entry-market authorized runtime scope is missing",
    )
    permitted_actions = _as_mapping(
        authorization.get("permitted_actions"),
        "entry-market permitted actions are missing",
    )
    prohibited_actions = _as_mapping(
        authorization.get("prohibited_actions"),
        "entry-market prohibited actions are missing",
    )
    preserved = _as_mapping(
        authorization.get("preserved_checkpoint_identity"),
        "entry-market preserved checkpoint identity is missing",
    )
    if (
        authorization_file.sha256
        != _EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AUTHORIZATION_SHA256
        or authorization_file.size_bytes != 7513
        or authorization.get("status")
        != "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT_AUTHORIZED"
        or authorization.get("authorization_id")
        != authorization_binding["authorization_id"]
        or auth_semantics.get("authorization_form")
        != authorization_binding["authorization_form"]
        or auth_semantics.get("source_message_id")
        != authorization_binding["source_message_id"]
        or auth_semantics.get("source_message_text")
        != authorization_binding["source_message_exact"]
        or auth_semantics.get("source_message_utf8_size_bytes") != 36
        or auth_semantics.get("source_message_has_terminal_newline") is not False
        or auth_semantics.get("source_message_text_sha256")
        != authorization_binding["source_message_sha256"]
        or auth_semantics.get("authorized") is not True
        or auth_semantics.get("current_v6_amendment_implementation_authorized_now")
        is not True
        or auth_semantics.get(
            "equivalent_evidence_backed_non_economic_repair_gates_until_stage13_completion_authorized"
        )
        is not True
        or auth_semantics.get("routine_non_economic_repair_gate_reprompt_required")
        is not False
        or standing.get("granted") is not True
        or standing.get("production_issue_stage13_unblind_approval_required")
        is not True
        or standing.get("production_require_stage13_unblind_authorization_required")
        is not True
        or standing.get("stage13_approval_receipt_issued_now") is not False
        or standing.get("stage13_execution_authorized_now") is not False
        or standing.get("current_v5_stage13_approval_valid_for_v6") is not False
        or standing.get("additional_user_prompt_after_v6_verification_required")
        is not False
        or standing.get("source_message_sha256_for_future_receipt")
        != authorization_binding["source_message_sha256"]
        or auth_limits.get("material_scientific_or_economic_rule_changes") is not False
        or auth_limits.get("destructive_actions") is not False
        or auth_limits.get(
            "approval_may_not_bypass_failed_tests_or_independent_verification"
        )
        is not True
        or auth_limits.get(
            "approval_may_not_bypass_exact_stage12_binding_authentication"
        )
        is not True
        or auth_limits.get("stop_if_scope_becomes_material_or_destructive") is not True
        or any(value is not True for value in permitted_actions.values())
        or any(value is not True for value in prohibited_actions.values())
        or authorized_runtime.get("entry_date") != "2025-12-01"
        or authorized_runtime.get("daily_retained_rows") != 10605
        or authorized_runtime.get("daily_canonical_rows") != 10596
        or authorized_runtime.get("daily_consumer_exact_duplicate_groups") != 9
        or authorized_runtime.get("daily_consumer_exact_duplicate_surplus_rows") != 9
        or authorized_runtime.get("daily_consumer_conflicting_duplicate_groups") != 0
        or authorized_runtime.get("active_shares_rows") != 10593
        or authorized_runtime.get("closed_tombstone_zero_match_rows") != 3
        or authorized_runtime.get("unexpected_zero_match_rows") != 0
        or authorized_runtime.get("tombstone_key_set_sha256")
        != "5af68213b30cf617bf7b96eac18f1d62a12ca0cda2f3e19a27c132996d7a4b66"
        or preserved.get("parent_context_sha256")
        != "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e"
        or preserved.get("checkpoint_receipt_count") != 431
        or preserved.get("parent_mutated") is not False
        or preserved.get("outcome_source_receipt_count") != 0
        or preserved.get("economic_checkpoint_count") != 0
    ):
        raise Stage12FreezeError("entry-market authorization semantics differ")

    blocker_safe = _as_mapping(
        blocker.get("safe_stop"), "entry-market blocker safe-stop state is missing"
    )
    blocker_preflight = _as_mapping(
        blocker.get("production_preflight"),
        "entry-market blocker preflight is missing",
    )
    blocker_root = _as_mapping(
        blocker.get("root_cause"), "entry-market blocker root cause is missing"
    )
    blocker_scope = _as_mapping(
        blocker.get("narrow_v6_scope"), "entry-market v6 scope is missing"
    )
    blocker_replacement = _as_mapping(
        blocker.get("replacement_authority"),
        "entry-market blocker replacement authority is missing",
    )
    blocker_sources = _as_mapping(
        blocker.get("authenticated_sources"),
        "entry-market blocker source identities are missing",
    )
    blocker_daily_source = _as_mapping(
        blocker_sources.get("daily"), "entry-market blocker daily source is missing"
    )
    blocker_shares_source = _as_mapping(
        blocker_sources.get("shares"), "entry-market blocker shares source is missing"
    )
    blocker_extends = _as_mapping(
        blocker.get("extends_prior_blocker"),
        "entry-market blocker prior-blocker binding is missing",
    )
    exact_diagnostic = _as_mapping(
        exact_blocker.get("exact_diagnostic"),
        "exact-duplicate diagnostic is missing",
    )
    exact_root = _as_mapping(
        exact_blocker.get("root_cause"),
        "exact-duplicate root cause is missing",
    )
    exact_required = _as_mapping(
        exact_blocker.get("required_correction"),
        "exact-duplicate correction contract is missing",
    )
    if (
        exact_blocker_file.sha256
        != "813f5f38a802eb0b926cdaca565016a27a40a4ac96b7e8cb0a89fc1f08bb3f46"
        or exact_blocker.get("status")
        != "BLOCKED_PENDING_NARROW_EXACT_DUPLICATE_CANONICALIZATION_AMENDMENT_AUTHORIZATION"
        or exact_blocker.get("classification")
        != "NON_ECONOMIC_ENTRY_MARKET_CANONICALIZATION_GUARD_CONTRADICTION"
        or exact_diagnostic.get("entry_date") != "2025-12-01"
        or exact_diagnostic.get("raw_retained_rows") != 10605
        or exact_diagnostic.get("exact_duplicate_groups") != 9
        or exact_diagnostic.get("exact_duplicate_surplus_rows") != 9
        or exact_diagnostic.get("conflicting_duplicate_groups") != 0
        or exact_diagnostic.get("canonical_output_rows") != 10596
        or exact_diagnostic.get("canonical_output_distinct_keys") != 10596
        or exact_diagnostic.get("canonical_output_return_columns_present") != []
        or exact_root.get("production_output_has_duplicate_keys") is not False
        or exact_root.get("raw_data_has_conflicting_records") is not False
        or exact_required.get("retain_conflicting_duplicate_rejection") is not True
        or exact_required.get("retain_exact_stage2_canonicalizer") is not True
        or exact_required.get("narrow_amendment_implementation_authorized_now")
        is not False
        or blocker_extends.get("path") != incident["exact_duplicate_blocker"]["path"]
        or blocker_extends.get("sha256") != exact_blocker_file.sha256
        or blocker_daily_source.get("sha256") != projection.get("daily_source_sha256")
        or blocker_daily_source.get("size_bytes")
        != projection.get("daily_source_size_bytes")
        or blocker_daily_source.get("full_file_rows")
        != projection.get("daily_source_rows")
        or blocker_shares_source.get("sha256") != projection.get("shares_source_sha256")
        or blocker_shares_source.get("size_bytes")
        != projection.get("shares_source_size_bytes")
        or blocker_shares_source.get("full_file_rows")
        != projection.get("shares_source_rows")
        or blocker_file.sha256
        != _EXPECTED_STAGE13_ENTRY_MARKET_PROJECTION_BLOCKER_SHA256
        or blocker.get("status") != incident["combined_projection_blocker"]["status"]
        or blocker.get("classification")
        != "NON_ECONOMIC_ENTRY_MARKET_PROJECTION_GUARD_CONTRADICTIONS"
        or blocker_safe.get("stage13_process_running") is not False
        or blocker_safe.get("launchd_service_loaded") is not False
        or blocker_safe.get("preserved_parent_mutated") is not False
        or blocker_safe.get("preserved_decision_checkpoint_count") != 431
        or blocker_safe.get("only_missing_decision_checkpoint")
        != "final_market_inputs/2025-12"
        or blocker_safe.get("failed_context_receipt_count") != 0
        or blocker_preflight.get("daily_retained_rows") != 10605
        or blocker_preflight.get("daily_exact_duplicate_key_groups") != 9
        or blocker_preflight.get("daily_exact_duplicate_surplus_rows") != 9
        or blocker_preflight.get("daily_conflicting_duplicate_groups") != 0
        or blocker_preflight.get("daily_canonical_rows") != 10596
        or blocker_preflight.get("active_shares_interval_rows") != 10593
        or blocker_preflight.get("daily_rows_without_active_shares") != 3
        or blocker_preflight.get("unexpected_daily_rows_without_active_shares") != 0
        or blocker_preflight.get("third_blocker_found") is not False
        or blocker_root.get("production_daily_output_has_duplicate_keys") is not False
        or blocker_root.get("raw_daily_source_has_conflicting_duplicate_records")
        is not False
        or blocker_scope.get("expected_inventory_delta")
        != {"added": 2, "changed": 7, "removed": 0}
        or blocker_scope.get("changed_project_files")
        != [
            "src/portfolio_bot/final_freeze/inventory.py",
            "src/portfolio_bot/final_freeze/publication.py",
            "src/portfolio_bot/final_freeze/verification.py",
            "src/portfolio_bot/final_test/final_source_replay.py",
            "src/portfolio_bot/validation/validation_market_extension.py",
            "tests/unit/test_final_source_builder.py",
            "tests/unit/test_stage12_final_freeze.py",
        ]
        or blocker_replacement.get("narrow_amendment_implementation_authorized_now")
        is not False
        or blocker_replacement.get("stage12_v6_replacement_freeze_required") is not True
        or blocker_replacement.get("current_stage13_v5_approval_valid_for_v6")
        is not False
        or blocker_replacement.get("stage13_retry_under_v5_forbidden") is not True
    ):
        raise Stage12FreezeError("entry-market blocker semantics differ")

    daily = _as_mapping(
        preflight.get("daily_projection"),
        "entry-market daily preflight is missing",
    )
    shares = _as_mapping(
        preflight.get("active_shares_projection"),
        "entry-market shares preflight is missing",
    )
    share_coverage = _as_mapping(
        preflight.get("share_coverage"),
        "entry-market coverage preflight is missing",
    )
    preflight_remaining = _as_mapping(
        preflight.get("remaining_predicates"),
        "entry-market remaining preflight predicates are missing",
    )
    preflight_sources = _as_mapping(
        preflight.get("sources"), "entry-market preflight sources are missing"
    )
    preflight_daily_source = _as_mapping(
        preflight_sources.get("daily"),
        "entry-market preflight daily source is missing",
    )
    preflight_shares_source = _as_mapping(
        preflight_sources.get("shares"),
        "entry-market preflight shares source is missing",
    )
    if (
        preflight_file.sha256 != _EXPECTED_STAGE13_ENTRY_MARKET_SHARE_PREFLIGHT_SHA256
        or preflight.get("status") != incident["share_coverage_preflight"]["status"]
        or preflight.get("entry_date") != "2025-12-01"
        or preflight_daily_source.get("sha256") != projection.get("daily_source_sha256")
        or preflight_daily_source.get("size_bytes")
        != projection.get("daily_source_size_bytes")
        or preflight_daily_source.get("full_file_rows")
        != projection.get("daily_source_rows")
        or preflight_shares_source.get("sha256")
        != projection.get("shares_source_sha256")
        or preflight_shares_source.get("size_bytes")
        != projection.get("shares_source_size_bytes")
        or preflight_shares_source.get("full_file_rows")
        != projection.get("shares_source_rows")
        or daily.get("requested_consumer_columns") != 29
        or daily.get("requested_return_columns") != 0
        or daily.get("retained_rows") != 10605
        or daily.get("consumer_exact_duplicate_key_groups") != 9
        or daily.get("consumer_exact_duplicate_surplus_rows") != 9
        or daily.get("consumer_conflicting_duplicate_groups") != 0
        or daily.get("canonical_rows") != 10596
        or daily.get("canonical_output_keys_unique") is not True
        or shares.get("active_interval_rows") != 10593
        or shares.get("duplicate_key_groups") != 0
        or shares.get("overlap_edges") != 0
        or shares.get("gap_edges") != 0
        or share_coverage.get("population_rows") != 10596
        or share_coverage.get("one_match_rows") != 10593
        or share_coverage.get("zero_match_rows") != 3
        or share_coverage.get("multiple_match_rows") != 0
        or share_coverage.get("unexpected_zero_match_rows") != 0
        or share_coverage.get("tombstone_rows") != 3
        or share_coverage.get("tombstone_predicate_id")
        != "stage13-entry-market-closed-delisting-tombstone-v1"
        or share_coverage.get("tombstone_key_set_sha256")
        != coverage.get("tombstone_key_set_sha256")
        or any(value is not True for value in preflight_remaining.values())
    ):
        raise Stage12FreezeError("entry-market share-coverage preflight differs")

    if (
        projection.get("entry_date") != "2025-12-01"
        or projection.get("daily_requested_consumer_column_count") != 29
        or projection.get("daily_requested_return_column_count") != 0
        or projection.get("daily_output_return_columns") != []
        or projection.get("final_market_inputs_inner_join_semantics_changed")
        is not False
        or canonicalization.get("daily_projection_retained_rows") != 10605
        or canonicalization.get("daily_canonical_rows") != 10596
        or canonicalization.get("daily_consumer_exact_duplicate_key_groups") != 9
        or canonicalization.get("daily_consumer_exact_duplicate_surplus_rows") != 9
        or canonicalization.get("daily_consumer_conflicting_duplicate_groups") != 0
        or canonicalization.get("daily_output_duplicate_keys") != 0
        or any(
            canonicalization.get(name) is not True
            for name in (
                "retained_equals_canonical_plus_duplicate_surplus",
                "duplicate_groups_must_not_exceed_surplus_rows",
                "duplicate_groups_zero_if_and_only_if_surplus_zero",
                "all_consumer_field_conflicts_remain_fatal",
                "canonical_output_keys_unique",
                "exact_duplicate_multiplicity_only_removed",
            )
        )
        or coverage.get("population_rows") != 10596
        or coverage.get("one_match_rows") != 10593
        or coverage.get("zero_match_rows") != 3
        or coverage.get("multiple_match_rows") != 0
        or coverage.get("unexpected_zero_match_rows") != 0
        or coverage.get("active_share_rows_without_daily") != 0
        or coverage.get("tombstone_rows") != 3
        or coverage.get("tombstone_predicate_id")
        != "stage13-entry-market-closed-delisting-tombstone-v1"
        or coverage.get("any_fourth_or_differently_shaped_zero_match_row_fatal")
        is not True
        or coverage.get("final_market_inputs_inner_join_unchanged") is not True
        or receipt_contract.get("all_counts_exact_nonnegative_integers") is not True
        or receipt_contract.get(
            "stable_reauthentication_recomputes_coverage_from_both_parquets"
        )
        is not True
        or receipt_contract.get("receipt_tamper_must_fail_closed") is not True
    ):
        raise Stage12FreezeError("entry-market projection contract differs")

    expected_transition = {
        "exact_changed_runtime_member_count": 2,
        "members": [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": _EXPECTED_ENTRY_MARKET_REPLAY_PRE_SHA256,
                "post_amendment_sha256": _EXPECTED_ENTRY_MARKET_REPLAY_POST_SHA256,
            },
            {
                "path": "src/portfolio_bot/validation/validation_market_extension.py",
                "pre_amendment_sha256": _EXPECTED_ENTRY_MARKET_EXTENSION_PRE_SHA256,
                "post_amendment_sha256": _EXPECTED_ENTRY_MARKET_EXTENSION_POST_SHA256,
            },
        ],
    }
    _require_exact_mapping(
        transition,
        expected_transition,
        label="entry-market runtime source transition",
    )
    for member in expected_transition["members"]:
        try:
            runtime_path = confined_relative_path(
                project_root,
                member["path"],
                file_required=True,
                label="entry-market repaired runtime member",
            )
            runtime_file = stable_file(runtime_path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "entry-market repaired runtime member path is unsafe"
            ) from exc
        current_sha256 = (
            _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
            if member["path"] == "src/portfolio_bot/final_test/final_source_replay.py"
            else member["post_amendment_sha256"]
        )
        if not _is_consolidated_v12(policy) and runtime_file.sha256 != current_sha256:
            raise Stage12FreezeError(
                f"entry-market repaired runtime member differs: {member['path']}"
            )

    prior_policy, prior_policy_file = bound_json(
        _as_mapping(prior.get("policy"), "prior v5 policy binding is missing"),
        label="prior v5 Stage 12 policy",
    )
    prior_inventory, prior_inventory_file = bound_json(
        _as_mapping(prior.get("file_inventory"), "prior v5 inventory is missing"),
        label="prior v5 Stage 12 file inventory",
    )
    prior_primary, prior_primary_file = bound_json(
        _as_mapping(
            prior.get("primary_pointer_archive"),
            "prior v5 primary pointer archive is missing",
        ),
        label="prior v5 Stage 12 primary pointer archive",
    )
    prior_verified, prior_verified_file = bound_json(
        _as_mapping(
            prior.get("verified_pointer_archive"),
            "prior v5 verified pointer archive is missing",
        ),
        label="prior v5 Stage 12 verified pointer archive",
    )
    prior_approval, prior_approval_file = bound_json(
        _as_mapping(
            prior.get("stage13_approval_archive"),
            "prior v5 Stage 13 approval archive is missing",
        ),
        label="prior v5 Stage 13 approval archive",
    )
    prior_approval_payload = dict(prior_approval)
    prior_approval_payload_sha = prior_approval_payload.pop("payload_sha256", None)
    if (
        prior_policy_file.sha256 != _EXPECTED_PRIOR_V5_REPLACEMENT_POLICY_SHA256
        or prior_policy.get("policy_id")
        != "stage12-post-unblind-december-market-input-scope-partition-replacement-freeze-policy-v5"
        or prior_inventory_file.sha256 != _PRIOR_V5_STAGE12_FILE_INVENTORY_SHA256
        or prior_inventory.get("file_count") != 600
        or prior_inventory.get("tree_sha256")
        != "7dd91fdcd41e16f80572debe2cf04b727b744f27ee9666026f64de0b15fd5806"
        or prior_primary_file.sha256
        != "5d8820eba667dee1f311811ad7e19224fa5b2d67078224df258765012d5c00fe"
        or prior_primary.get("freeze_id") != "stage12-freeze-fce138650a3857d90f0b65a2"
        or prior_verified_file.sha256
        != "aed589c5d3740337edc8f4298abc72ec0f9971fc0c0cb3aac289b61abd3b4028"
        or prior_verified.get("verification_id")
        != "stage12-verify-2f1cf9143445578751ad3487"
        or prior_verified.get("primary_pointer_sha256") != prior_primary_file.sha256
        or prior_approval_file.sha256
        != "dc856b1954e7d7779a43e7c0b7ecf0102f9a7fd8121f2bd3594d6567543283fa"
        or prior_approval.get("approved") is not True
        or prior_approval.get("stage12_binding_sha256")
        != "db522ddf257ed117b8be2af4d0eb0d79e916027b42408475f42557cc0d69838e"
        or prior_approval_payload_sha
        != "b036d0506d96d98cd67ba342a1e642f4102562e53f7862cd54660874b6a60349"
        or sha256_bytes(canonical_json_bytes(prior_approval_payload))
        != prior_approval_payload_sha
    ):
        raise Stage12FreezeError("prior v5 Stage 12 authority semantics differ")

    diagnostic = _as_mapping(
        boundary.get("prior_diagnostic_access_disclosure"),
        "entry-market prior diagnostic disclosure is missing",
    )
    if (
        any(
            boundary.get(name) != 0
            for name in (
                "guarded_stage12_publication_and_verification_raw_payload_files_opened",
                "guarded_stage12_parent_checkpoint_payload_files_opened",
                "guarded_stage12_final_period_payload_files_opened",
                "guarded_stage12_final_period_rows_read",
                "guarded_stage12_outcome_or_economic_payload_files_opened",
                "guarded_stage12_economic_or_outcome_months_executed",
                "guarded_stage12_final_forecasts_weights_trades_or_pnl_created",
            )
        )
        or boundary.get("failed_context_or_preserved_parent_mutated") is not False
        or boundary.get(
            "calendar_dates_schemas_keys_features_models_or_economic_rules_changed"
        )
        is not False
        or boundary.get("scientific_or_economic_result_informed_choice") is not False
        or diagnostic.get("both_pinned_raw_files_freshly_sha256_authenticated")
        is not True
        or diagnostic.get("all_daily_and_shares_rows_scanned") is not True
        or diagnostic.get("daily_return_fields_decoded_or_aggregated") is not False
        or diagnostic.get("independent_diagnostic_included_four_raw_return_columns")
        is not True
        or diagnostic.get("parent_predictor_or_checkpoint_payloads_opened") is not False
        or diagnostic.get("outcome_accounting_or_economic_payloads_opened") is not False
        or diagnostic.get("economic_or_model_choice_informed") is not False
        or replacement
        != {
            "stage12_v6_replacement_freeze_required": True,
            "independent_stage12_v6_verification_required": True,
            "current_stage13_v5_approval_valid_for_v6": False,
            "standing_stage13_approval_granted": True,
            "fresh_binding_specific_stage13_receipt_required_after_v6": True,
            "fresh_binding_specific_stage13_receipt_issued_by_this_amendment": False,
            "additional_user_prompt_after_v6_verification_required": False,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_execution_authorized_now": False,
            "stage13_retry_under_v5_forbidden": True,
        }
        or tests.get("required_files")
        != [
            "tests/unit/test_final_source_builder.py",
            "tests/unit/test_stage12_final_freeze.py",
        ]
        or any(tests.get(name) is not True for name in set(tests) - {"required_files"})
    ):
        raise Stage12FreezeError("entry-market governance boundary differs")

    return amendment, {
        "relative_path": expected_binding["path"],
        "sha256": amendment_file.sha256,
        "size_bytes": amendment_file.size_bytes,
        "status": amendment["status"],
        "amendment_id": amendment["amendment_id"],
        "authorization_receipt": {
            "relative_path": authorization_binding["path"],
            "sha256": authorization_file.sha256,
            "size_bytes": authorization_file.size_bytes,
            "status": authorization["status"],
            "authorization_id": authorization["authorization_id"],
            "standing_stage13_approval": dict(standing),
        },
        "combined_projection_blocker_receipt": {
            "relative_path": incident["combined_projection_blocker"]["path"],
            "sha256": blocker_file.sha256,
            "size_bytes": blocker_file.size_bytes,
            "status": blocker["status"],
            "classification": blocker["classification"],
        },
        "share_coverage_preflight_receipt": {
            "relative_path": incident["share_coverage_preflight"]["path"],
            "sha256": preflight_file.sha256,
            "size_bytes": preflight_file.size_bytes,
            "status": preflight["status"],
        },
        "prior_authority_receipts": {
            "policy": {
                "relative_path": prior["policy"]["path"],
                "sha256": prior_policy_file.sha256,
                "size_bytes": prior_policy_file.size_bytes,
            },
            "file_inventory": {
                "relative_path": prior["file_inventory"]["path"],
                "sha256": prior_inventory_file.sha256,
                "size_bytes": prior_inventory_file.size_bytes,
            },
            "primary_pointer_archive": {
                "relative_path": prior["primary_pointer_archive"]["path"],
                "sha256": prior_primary_file.sha256,
                "size_bytes": prior_primary_file.size_bytes,
            },
            "verified_pointer_archive": {
                "relative_path": prior["verified_pointer_archive"]["path"],
                "sha256": prior_verified_file.sha256,
                "size_bytes": prior_verified_file.size_bytes,
            },
            "stage13_approval_archive": {
                "relative_path": prior["stage13_approval_archive"]["path"],
                "sha256": prior_approval_file.sha256,
                "size_bytes": prior_approval_file.size_bytes,
            },
        },
        "entry_market_projection_contract": dict(projection),
        "exact_duplicate_canonicalization_contract": dict(canonicalization),
        "closed_tombstone_share_coverage_contract": dict(coverage),
        "receipt_contract": dict(receipt_contract),
        "source_transition": dict(transition),
        "non_economic_boundary": dict(boundary),
        "replacement_authority": dict(replacement),
    }


def _validate_post_unblind_accounting_missing_execution_cost_transport_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Authenticate the narrow rule-restoring Stage 13 accounting transport."""

    expected_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_accounting_missing_execution_cost_transport_"
            "amendment_v1.json"
        ),
        "sha256": _EXPECTED_ACCOUNTING_TRANSPORT_AMENDMENT_SHA256,
    }
    binding = _as_mapping(
        policy.get(
            "post_unblind_accounting_missing_execution_cost_transport_amendment"
        ),
        "accounting missing-input transport amendment binding is missing",
    )
    _require_exact_mapping(
        binding,
        expected_binding,
        label="accounting missing-input transport amendment binding",
    )

    def bound_json(
        raw: Mapping[str, Any],
        *,
        expected_sha256: str,
        expected_status: str,
        label: str,
        binding_status_required: bool = True,
    ) -> tuple[Mapping[str, Any], StableFile]:
        try:
            path = confined_relative_path(
                project_root,
                raw.get("path"),
                file_required=True,
                label=label,
            )
            value, record = load_json_object(path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if (
            raw.get("sha256") != expected_sha256
            or record.sha256 != expected_sha256
            or (binding_status_required and raw.get("status") != expected_status)
            or value.get("status") != expected_status
        ):
            raise Stage12FreezeError(f"{label} identity differs")
        return value, record

    amendment, amendment_record = bound_json(
        binding,
        expected_sha256=_EXPECTED_ACCOUNTING_TRANSPORT_AMENDMENT_SHA256,
        expected_status=(
            "STAGE12_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT_"
            "AUTHORIZED"
        ),
        label="accounting missing-input transport amendment",
        binding_status_required=False,
    )
    if (
        set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization_receipt",
            "materiality_policy",
            "prior_v7_authority",
            "incident_authorities",
            "diagnosis",
            "transport_contract",
            "source_transition",
            "focused_test_evidence",
            "access_disclosure",
            "checkpoint_reuse",
            "non_economic_boundary",
            "replacement_authority",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-accounting-missing-execution-cost-transport-v1"
        or amendment.get("classification")
        != (
            "non_economic_rule_restoring_stage13_transport_correction_at_an_"
            "economic_safety_boundary"
        )
    ):
        raise Stage12FreezeError("accounting missing-input amendment semantics differ")

    authorization, authorization_record = bound_json(
        _as_mapping(
            amendment.get("authorization_receipt"),
            "accounting transport authorization is missing",
        ),
        expected_sha256=_EXPECTED_ACCOUNTING_TRANSPORT_AUTHORIZATION_SHA256,
        expected_status=(
            "STAGE12_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT_"
            "AUTHORIZED"
        ),
        label="accounting transport authorization",
    )
    materiality, materiality_record = bound_json(
        _as_mapping(
            amendment.get("materiality_policy"),
            "thesis error-materiality policy is missing",
        ),
        expected_sha256=_EXPECTED_THESIS_ERROR_MATERIALITY_POLICY_SHA256,
        expected_status="THESIS_ERROR_MATERIALITY_POLICY_ACTIVE",
        label="thesis error-materiality policy",
    )
    incidents = _as_mapping(
        amendment.get("incident_authorities"),
        "accounting transport incident authorities are missing",
    )
    blocker, blocker_record = bound_json(
        _as_mapping(incidents.get("blocker"), "ADV blocker binding is missing"),
        expected_sha256=_EXPECTED_STAGE13_ADV20_NULLABILITY_BLOCKER_SHA256,
        expected_status=(
            "STAGE13_FAILED_CLOSED_BEFORE_FIRST_OUTCOME_MONTH_ON_ADV20_"
            "NULLABILITY_CONTRADICTION"
        ),
        label="Stage 13 ADV nullability blocker",
    )
    preflight, preflight_record = bound_json(
        _as_mapping(
            incidents.get("preflight"),
            "accounting missing-input preflight binding is missing",
        ),
        expected_sha256=_EXPECTED_STAGE13_ACCOUNTING_PREFLIGHT_SHA256,
        expected_status=(
            "STAGE13_ACCOUNTING_MISSING_EXECUTION_COST_PREFLIGHT_COMPLETE"
        ),
        label="Stage 13 accounting missing-input preflight",
    )

    authorization_scope = _as_mapping(
        authorization.get("authorization"),
        "accounting transport authorization scope is missing",
    )
    materiality_rule = _as_mapping(
        materiality.get("decision_rule"),
        "thesis materiality decision rule is missing",
    )
    blocker_rule = _as_mapping(
        blocker.get("materiality_rule"),
        "ADV blocker materiality rule is missing",
    )
    diagnosis = _as_mapping(
        amendment.get("diagnosis"),
        "accounting missing-input diagnosis is missing",
    )
    preflight_diagnosis = _as_mapping(
        preflight.get("accounting_security_month_domain"),
        "accounting preflight domain is missing",
    )
    transport = _as_mapping(
        amendment.get("transport_contract"),
        "accounting transport contract is missing",
    )
    access = _as_mapping(
        amendment.get("access_disclosure"),
        "accounting transport access disclosure is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "accounting transport non-economic boundary is missing",
    )
    checkpoint = _as_mapping(
        amendment.get("checkpoint_reuse"),
        "accounting transport checkpoint contract is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "accounting transport replacement authority is missing",
    )
    if (
        authorization_scope.get("authorized") is not True
        or authorization_scope.get("routine_non_economic_repair_gate_reprompt_required")
        is not False
        or authorization_scope.get("stage13_execution_authorized_now") is not False
        or not {
            "economically_material",
            "negligible_non_economic",
            "uncertain_boundary",
        }.issubset(materiality_rule)
        or blocker_rule.get("direct_exception_or_bypass_permitted") is not False
        or diagnosis
        != {
            "accounting_security_month_rows": 232034,
            "missing_entry_close_rows": 27297,
            "missing_full_spread_rows": 0,
            "missing_adv_rows": 25,
            "missing_adv_distinct_permnos": 23,
            "missing_impact_volatility_rows": 0,
            "missing_source_borrow_rows": 118596,
            "short_eligible_missing_source_borrow_rows": 0,
            "all_missing_adv_rows_authenticated_nontradable": True,
            "stage8_or_stage11_repair_required": False,
            "model_or_portfolio_rule_change_required": False,
        }
        or preflight_diagnosis.get("active_security_months") != 232034
        or preflight_diagnosis.get("missing_adv_security_months") != 25
        or preflight_diagnosis.get("missing_adv_distinct_permnos") != 23
        or preflight_diagnosis.get(
            "short_eligible_missing_primary_borrow_security_months"
        )
        != 0
        or transport.get("input_registry_sha256")
        != "bfef5266ec13af8377d3950fb70534b1ade27b599797cd72bbf315578b10994c"
        or transport.get("input_registry_unchanged") is not True
        or transport.get("nan_transport_fields")
        != [
            "entry_close_usd_per_share",
            "full_spread_20d",
            "adv20_usd",
            "impact_volatility_daily",
        ]
        or transport.get("missing_accounting_borrow_default_annual") != 0.01
        or transport.get("non_null_source_borrow_required_annual") != 0.01
        or transport.get("positive_adv_imputation_permitted") is not False
        or transport.get("row_drop_permitted") is not False
        or transport.get("missing_execution_or_cost_trade_capacity") != 0
        or transport.get("exact_unchanged_holding_with_missing_input_permitted")
        is not True
        or transport.get("nonzero_trade_with_missing_input_permitted") is not False
        or access.get("prior_stage13_read_only_diagnostic_parquet_files_opened") != 290
        or access.get("stage12_v8_parent_checkpoint_payload_files_opened_required") != 0
        or access.get("stage12_v8_outcome_or_economic_payload_files_opened_required")
        != 0
        or boundary.get("rows_dropped") != 0
        or boundary.get("positive_adv_imputations") != 0
        or boundary.get("protected_payloads_opened_by_stage12_v8") != 0
        or checkpoint.get("authenticated_checkpoint_pairs_rebound") != 431
        or checkpoint.get("only_rebuilt_decision_checkpoint")
        != "final_market_inputs/2025-12"
        or checkpoint.get("direct_432_rebind_protocol_required") is not False
        or replacement.get("replacement_stage12_v8_identity_required") is not True
        or replacement.get("independent_stage12_verification_required") is not True
        or replacement.get("prior_v7_stage13_approval_valid_for_v8") is not False
        or replacement.get("additional_user_prompt_after_v8_verification_required")
        is not False
    ):
        raise Stage12FreezeError("accounting missing-input contract differs")

    prior = _as_mapping(
        amendment.get("prior_v7_authority"),
        "prior v7 authority is missing",
    )
    expected_prior_files = {
        str(prior.get("primary_pointer_archive_path")): (
            "6f4680f06b3ba773e4ba814122778675e255e60a1d9d22805012112214c2749c"
        ),
        str(prior.get("verified_pointer_archive_path")): (
            "761a7a114ed1aea2feb2bc3b9a0dbec12e003651d2b6c352f8194ac47b2204fc"
        ),
        str(prior.get("stage13_approval_archive_path")): (
            "b53bb143fa03ffe93634b07211ad40ca92ff0436b7f7f42abd4c6d04dde16c51"
        ),
        str(prior.get("file_inventory_path")): (
            _PRIOR_V7_STAGE12_FILE_INVENTORY_SHA256
        ),
        str(prior.get("policy_path")): _EXPECTED_PRIOR_V7_REPLACEMENT_POLICY_SHA256,
    }
    if (
        prior.get("file_count") != 604
        or prior.get("project_tree_sha256")
        != "f94c50b9d94f7ed88fd09081a95f75c04006b1d89c79fdd1dcdb2b81886e640c"
        or prior.get("freeze_id") != "stage12-freeze-58abf3e871270249636fd180"
        or prior.get("manifest_sha256")
        != "18acebcbc1fe5fa12d3013338cf8a85a44533b2a64aff795c00d5e9db3506d09"
        or prior.get("stage12_binding_sha256")
        != "f94ddb98873c8626fb20f0d8921e6624262ae4f4619710f8fe54567c3b390ff7"
        or prior.get("stage13_approval_payload_sha256")
        != "0f4aaf8770226b3559e8af9fe946a2f0a7fc36a3f0e79188c70d3f3471fd0670"
    ):
        raise Stage12FreezeError("prior v7 accounting authority differs")
    for relative_path, expected_sha in expected_prior_files.items():
        try:
            selected = confined_relative_path(
                project_root,
                relative_path,
                file_required=True,
                label="prior v7 accounting authority",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("prior v7 authority path is unsafe") from exc
        if record.sha256 != expected_sha:
            raise Stage12FreezeError("prior v7 authority identity differs")

    transition = _as_mapping(
        amendment.get("source_transition"),
        "accounting transport source transition is missing",
    )
    members = transition.get("members")
    expected_members = [
        {
            "path": "src/portfolio_bot/final_test/final_source_replay.py",
            "pre_amendment_sha256": _EXPECTED_ACCOUNTING_REPLAY_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_ACCOUNTING_REPLAY_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/input_materialization.py",
            "pre_amendment_sha256": _EXPECTED_ACCOUNTING_MATERIALIZER_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_ACCOUNTING_MATERIALIZER_POST_SHA256,
        },
    ]
    if (
        transition.get("exact_changed_runtime_member_count") != 2
        or members != expected_members
    ):
        raise Stage12FreezeError("accounting transport source transition differs")
    for member in expected_members:
        try:
            selected = confined_relative_path(
                project_root,
                member["path"],
                file_required=True,
                label="accounting transport runtime member",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("accounting runtime path is unsafe") from exc
        expected_current_sha = (
            _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
            if member["path"] == "src/portfolio_bot/final_test/final_source_replay.py"
            else member["post_amendment_sha256"]
        )
        if not _is_consolidated_v12(policy) and record.sha256 != expected_current_sha:
            raise Stage12FreezeError("accounting runtime source identity differs")

    return amendment, {
        "relative_path": (
            POST_UNBLIND_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT.relative_to(
                project_root
            ).as_posix()
        ),
        "sha256": amendment_record.sha256,
        "size_bytes": amendment_record.size_bytes,
        "authorization": {
            "sha256": authorization_record.sha256,
            "size_bytes": authorization_record.size_bytes,
        },
        "materiality_policy": {
            "sha256": materiality_record.sha256,
            "size_bytes": materiality_record.size_bytes,
        },
        "blocker": {
            "sha256": blocker_record.sha256,
            "size_bytes": blocker_record.size_bytes,
        },
        "preflight": {
            "sha256": preflight_record.sha256,
            "size_bytes": preflight_record.size_bytes,
        },
    }


def _validate_post_unblind_logical_fingerprint_transport_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Authenticate the non-economic Arrow logical-identity correction."""

    expected_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_logical_fingerprint_transport_amendment_v1.json"
        ),
        "sha256": _EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SHA256,
    }
    binding = _as_mapping(
        policy.get("post_unblind_logical_fingerprint_transport_amendment"),
        "logical-fingerprint transport amendment binding is missing",
    )
    _require_exact_mapping(
        binding,
        expected_binding,
        label="logical-fingerprint transport amendment binding",
    )

    def bound_json(
        raw: Mapping[str, Any],
        *,
        expected_sha256: str,
        expected_status: str,
        label: str,
        binding_status_required: bool = True,
    ) -> tuple[Mapping[str, Any], StableFile]:
        try:
            path = confined_relative_path(
                project_root,
                raw.get("path"),
                file_required=True,
                label=label,
            )
            value, record = load_json_object(path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if (
            raw.get("sha256") != expected_sha256
            or record.sha256 != expected_sha256
            or (binding_status_required and raw.get("status") != expected_status)
            or value.get("status") != expected_status
        ):
            raise Stage12FreezeError(f"{label} identity differs")
        return value, record

    amendment, amendment_record = bound_json(
        binding,
        expected_sha256=_EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SHA256,
        expected_status="STAGE12_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_AUTHORIZED",
        label="logical-fingerprint transport amendment",
        binding_status_required=False,
    )
    if (
        set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization_receipt",
            "materiality_policy",
            "prior_v8_authority",
            "incident_authorities",
            "diagnosis",
            "logical_comparator_contract",
            "source_transition",
            "focused_test_evidence",
            "access_disclosure",
            "checkpoint_reuse",
            "non_economic_boundary",
            "replacement_authority",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-logical-fingerprint-transport-v1"
        or amendment.get("classification")
        != "proven_non_economic_arrow_physical_buffer_false_difference_correction"
        or sha256_bytes(canonical_json_bytes(amendment))
        != _EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SEMANTIC_SHA256
    ):
        raise Stage12FreezeError("logical-fingerprint amendment semantics differ")

    authorization, authorization_record = bound_json(
        _as_mapping(
            amendment.get("authorization_receipt"),
            "logical-fingerprint transport authorization is missing",
        ),
        expected_sha256=_EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AUTHORIZATION_SHA256,
        expected_status="STAGE12_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_AUTHORIZED",
        label="logical-fingerprint transport authorization",
    )
    materiality, materiality_record = bound_json(
        _as_mapping(
            amendment.get("materiality_policy"),
            "thesis error-materiality policy is missing",
        ),
        expected_sha256=_EXPECTED_THESIS_ERROR_MATERIALITY_POLICY_SHA256,
        expected_status="THESIS_ERROR_MATERIALITY_POLICY_ACTIVE",
        label="thesis error-materiality policy",
    )
    incidents = _as_mapping(
        amendment.get("incident_authorities"),
        "logical-fingerprint incident authorities are missing",
    )
    blocker, blocker_record = bound_json(
        _as_mapping(
            incidents.get("blocker"),
            "logical-fingerprint blocker binding is missing",
        ),
        expected_sha256=_EXPECTED_STAGE13_LOGICAL_FINGERPRINT_BLOCKER_SHA256,
        expected_status=(
            "STAGE13_FAILED_CLOSED_BEFORE_DECISION_BARRIER_ON_PHYSICAL_"
            "FINGERPRINT_FALSE_DIFFERENCE"
        ),
        label="Stage 13 logical-fingerprint blocker",
    )
    preflight, preflight_record = bound_json(
        _as_mapping(
            incidents.get("preflight"),
            "logical-fingerprint preflight binding is missing",
        ),
        expected_sha256=_EXPECTED_STAGE13_LOGICAL_FINGERPRINT_PREFLIGHT_SHA256,
        expected_status=(
            "STAGE13_FINAL_MARKET_INPUT_LOGICAL_FINGERPRINT_PREFLIGHT_COMPLETE"
        ),
        label="Stage 13 logical-fingerprint preflight",
    )

    authorization_scope = _as_mapping(
        authorization.get("authorization"),
        "logical-fingerprint authorization scope is missing",
    )
    required_safety = _as_mapping(
        authorization.get("required_safety"),
        "logical-fingerprint authorization safety contract is missing",
    )
    diagnosis = _as_mapping(
        amendment.get("diagnosis"),
        "logical-fingerprint diagnosis is missing",
    )
    comparator = _as_mapping(
        amendment.get("logical_comparator_contract"),
        "logical comparator contract is missing",
    )
    access = _as_mapping(
        amendment.get("access_disclosure"),
        "logical-fingerprint access disclosure is missing",
    )
    checkpoint = _as_mapping(
        amendment.get("checkpoint_reuse"),
        "logical-fingerprint checkpoint contract is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "logical-fingerprint non-economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "logical-fingerprint replacement authority is missing",
    )
    focused_tests = _as_mapping(
        amendment.get("focused_test_evidence"),
        "logical-fingerprint focused-test evidence is missing",
    )
    if (
        authorization_scope.get("authorized") is not True
        or authorization_scope.get("routine_non_economic_repair_gate_reprompt_required")
        is not False
        or authorization_scope.get("stage13_execution_authorized_now") is not False
        or required_safety.get("schema_must_match_with_metadata") is not True
        or required_safety.get("row_and_column_counts_must_match") is not True
        or required_safety.get("null_masks_must_match") is not True
        or required_safety.get("finite_float_bits_must_match") is not True
        or required_safety.get("signed_zero_bits_must_match") is not True
        or required_safety.get("nan_positions_must_match") is not True
        or required_safety.get("nan_payload_bits_may_be_canonicalized") is not True
        or required_safety.get(
            "persisted_parquet_and_receipt_hash_authentication_retained"
        )
        is not True
        or required_safety.get(
            "rows_values_dates_models_weights_trades_costs_returns_and_pnl_may_change"
        )
        is not False
        or diagnosis
        != {
            "table": "final_market_inputs",
            "holding_month": "2025-12",
            "rows": 3056,
            "columns": 60,
            "schema_equal_with_metadata": True,
            "arrow_table_logically_equal": True,
            "differing_logical_value_count": 0,
            "differing_null_mask_count": 0,
            "differing_finite_float_bit_count": 0,
            "nullable_hidden_value_buffer_difference_column_count": 2,
            "nullable_hidden_value_buffer_difference_columns": [
                "primary_borrow_rate_annual",
                "stress_borrow_rate_annual",
            ],
            "null_count_per_differing_column": 1379,
            "redundant_all_valid_bitmap_column_count": 53,
            "redundant_all_valid_bitmap_bytes_per_column": 382,
            "outcome_sources_opened": False,
        }
        or comparator
        != {
            "application_scope": (
                "only_in_memory_built_fragment_vs_parquet_reopened_staged_fragment"
            ),
            "schema_with_metadata_exact": True,
            "row_and_column_counts_exact": True,
            "nonfloating_arrow_values_and_nulls_exact": True,
            "floating_null_masks_exact": True,
            "finite_floating_point_bits_exact": True,
            "signed_zero_bits_exact": True,
            "infinity_bits_exact": True,
            "nan_positions_exact": True,
            "nan_payload_bits_canonicalized_only": True,
            "undefined_value_buffer_bytes_beneath_nulls_ignored": True,
            "persisted_receipt_logical_content_fingerprint_algorithm_changed": False,
            "persisted_parquet_sha256_authentication_changed": False,
            "checkpoint_receipt_schema_changed": False,
            "rows_values_dates_keys_formulas_models_weights_trades_costs_returns_pnl_or_inference_changed": False,
        }
        or preflight.get("table") != "final_market_inputs"
        or preflight.get("holding_month") != "2025-12"
        or preflight.get("rows") != 3056
        or preflight.get("columns") != 60
        or preflight.get("schema_equal_with_metadata") is not True
        or preflight.get("arrow_table_logically_equal") is not True
        or preflight.get("differing_logical_value_count") != 0
        or preflight.get("differing_null_mask_count") != 0
        or preflight.get("differing_finite_float_bit_count") != 0
        or preflight.get("economically_meaningful_difference") is not False
        or blocker.get("materiality_classification")
        != (
            "PROVEN_NEGLIGIBLE_NON_ECONOMIC_ARROW_PHYSICAL_BUFFER_IDENTITY_"
            "FALSE_DIFFERENCE"
        )
        or blocker.get("direct_bypass_permitted") is not False
        or access.get("authorized_decision_side_diagnostic_replays") != 1
        or access.get("diagnosed_decision_table_months") != 1
        or access.get("outcome_or_economic_payload_files_opened") != 0
        or access.get("models_fit") != 0
        or access.get("forecasts_weights_trades_or_pnl_created") != 0
        or access.get("stage12_v9_parent_checkpoint_payload_files_opened_required") != 0
        or access.get("stage12_v9_outcome_or_economic_payload_files_opened_required")
        != 0
        or checkpoint.get("stage13_decision_checkpoint_rebind_v2_unchanged") is not True
        or checkpoint.get("authenticated_checkpoint_pairs_rebound") != 431
        or checkpoint.get("only_rebuilt_decision_checkpoint")
        != "final_market_inputs/2025-12"
        or checkpoint.get("decision_barrier_recommitted_after_authentication")
        is not True
        or checkpoint.get("failed_v8_context_preserved_immutably") is not True
        or checkpoint.get("outcome_or_economic_checkpoint_reuse_available") is not False
        or boundary.get(
            "sample_rows_dates_features_models_candidates_weights_trades_cost_formulas_returns_pnl_or_inference_changed"
        )
        is not False
        or boundary.get("decision_barrier_committed_by_failed_v8_run") is not False
        or boundary.get("outcome_sources_opened_by_failed_v8_run") is not False
        or boundary.get("outcome_month_checkpoints") != 0
        or boundary.get("economic_month_checkpoints") != 0
        or boundary.get("stage12_analysis_or_model_training_repeated") is not False
        or replacement.get("replacement_stage12_v9_identity_required") is not True
        or replacement.get("independent_stage12_verification_required") is not True
        or replacement.get("prior_v8_stage13_approval_valid_for_v9") is not False
        or replacement.get("fresh_exact_binding_stage13_receipt_required") is not True
        or replacement.get("standing_approval_permits_receipt_after_verification")
        is not True
        or replacement.get("additional_user_prompt_after_v9_verification_required")
        is not False
        or replacement.get("stage13_execution_authorized_now") is not False
        or focused_tests.get("test_file") != "tests/unit/test_final_source_builder.py"
        or focused_tests.get("pre_amendment_sha256")
        != "7acc6a7ac4b5ab2af1919ba70279d9355649fc0998120e793099ddecfcbd6e39"
        or focused_tests.get("post_amendment_sha256")
        != "828d86021a4b511c1a892817ef56dd56f91b0b41f0cf558ab2cd5ca757755824"
        or focused_tests.get("focused_tests_passed") != 4
    ):
        raise Stage12FreezeError("logical-fingerprint transport contract differs")

    prior = _as_mapping(
        amendment.get("prior_v8_authority"),
        "prior v8 authority is missing",
    )
    expected_prior_files = {
        str(prior.get("primary_pointer_archive_path")): (
            "1a2314c978d92222e80d9a625fd2fecb692243dc7a964f5ead530c22bb6e4048"
        ),
        str(prior.get("verified_pointer_archive_path")): (
            "979d51cff41b05e3e6e32880ac91ba54f35c196eade63f2fce2483da70c7eb9c"
        ),
        str(prior.get("stage13_approval_archive_path")): (
            "5bafb96b36c7fc0e02ba44e4c607270a24c5d836c520a037b021506084dd1d4a"
        ),
        str(prior.get("file_inventory_path")): (
            _PRIOR_V8_STAGE12_FILE_INVENTORY_SHA256
        ),
        str(prior.get("policy_path")): _EXPECTED_PRIOR_V8_REPLACEMENT_POLICY_SHA256,
    }
    if (
        prior.get("file_count") != 606
        or prior.get("project_tree_sha256")
        != "83b79290c60f0b2b39d92eb783bf419bc8f268bd9612d1e1922aa2786b28e2b5"
        or prior.get("freeze_id") != "stage12-freeze-0e8c91f5940398fdf0fb9dba"
        or prior.get("manifest_sha256")
        != "1855ce20d754ad9d853f6acb57b5d0f34d3e1dd3a7775469118d982156629b16"
        or prior.get("primary_success_sha256")
        != "09e8ff855bbe74546bc65e1de7b86d4276897d51a58d281bd528209bcf0e3adb"
        or prior.get("verification_id") != "stage12-verify-abfbc7074602e93da006a218"
        or prior.get("verification_report_sha256")
        != "0e407fa860ec276169d3171a5d939460cef04b9d1daad1aa7632852413446662"
        or prior.get("verification_success_sha256")
        != "9689efa2442a9ded1f553c9737add3004b43d95939c40593ae49025876ecb427"
        or prior.get("stage12_binding_sha256")
        != "10f8f05e55b62ca0cb1a2eaeb46dd16d6b4ae1bdd3190a2daa9ab72e6d3d3e0a"
        or prior.get("stage13_approval_payload_sha256")
        != "c33ed879b36756fb91797f8f3ab20a5752581e35ffae71c4f6760b57b2ee2f27"
    ):
        raise Stage12FreezeError("prior v8 logical-fingerprint authority differs")
    for relative_path, expected_sha in expected_prior_files.items():
        try:
            selected = confined_relative_path(
                project_root,
                relative_path,
                file_required=True,
                label="prior v8 logical-fingerprint authority",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("prior v8 authority path is unsafe") from exc
        if record.sha256 != expected_sha:
            raise Stage12FreezeError("prior v8 authority identity differs")

    transition = _as_mapping(
        amendment.get("source_transition"),
        "logical-fingerprint source transition is missing",
    )
    expected_members = [
        {
            "path": "src/portfolio_bot/final_test/final_source_replay.py",
            "pre_amendment_sha256": _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256,
        }
    ]
    if (
        transition.get("exact_changed_runtime_member_count") != 1
        or transition.get("members") != expected_members
    ):
        raise Stage12FreezeError("logical-fingerprint source transition differs")
    try:
        selected = confined_relative_path(
            project_root,
            expected_members[0]["path"],
            file_required=True,
            label="logical-fingerprint runtime member",
        )
        runtime_record = stable_file(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("logical-fingerprint runtime path is unsafe") from exc
    if (
        not _is_consolidated_v12(policy)
        and runtime_record.sha256 != _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
    ):
        raise Stage12FreezeError("logical-fingerprint runtime source identity differs")

    return amendment, {
        "relative_path": (
            POST_UNBLIND_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT.relative_to(
                project_root
            ).as_posix()
        ),
        "sha256": amendment_record.sha256,
        "size_bytes": amendment_record.size_bytes,
        "authorization": {
            "sha256": authorization_record.sha256,
            "size_bytes": authorization_record.size_bytes,
        },
        "materiality_policy": {
            "sha256": materiality_record.sha256,
            "size_bytes": materiality_record.size_bytes,
        },
        "blocker": {
            "sha256": blocker_record.sha256,
            "size_bytes": blocker_record.size_bytes,
        },
        "preflight": {
            "sha256": preflight_record.sha256,
            "size_bytes": preflight_record.size_bytes,
        },
    }


def _validate_post_unblind_factor_covariance_ordinal_basis_alignment_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Authenticate the non-economic zero-based covariance ordinal alignment."""

    expected_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_factor_covariance_ordinal_basis_alignment_"
            "amendment_v1.json"
        ),
        "sha256": (
            _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_SHA256
        ),
    }
    binding = _as_mapping(
        policy.get("post_unblind_factor_covariance_ordinal_basis_alignment_amendment"),
        "factor-covariance ordinal-basis amendment binding is missing",
    )
    _require_exact_mapping(
        binding,
        expected_binding,
        label="factor-covariance ordinal-basis amendment binding",
    )

    def bound_json(
        raw: Mapping[str, Any],
        *,
        expected_sha256: str,
        expected_status: str | None,
        label: str,
        binding_status_required: bool = True,
    ) -> tuple[Mapping[str, Any], StableFile]:
        try:
            path = confined_relative_path(
                project_root,
                raw.get("path"),
                file_required=True,
                label=label,
            )
            value, record = load_json_object(path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if (
            raw.get("sha256") != expected_sha256
            or record.sha256 != expected_sha256
            or (
                expected_status is not None
                and binding_status_required
                and raw.get("status") != expected_status
            )
            or (expected_status is not None and value.get("status") != expected_status)
        ):
            raise Stage12FreezeError(f"{label} identity differs")
        return value, record

    amendment, amendment_record = bound_json(
        binding,
        expected_sha256=(
            _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_SHA256
        ),
        expected_status=(
            "STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_" "AUTHORIZED"
        ),
        label="factor-covariance ordinal-basis amendment",
        binding_status_required=False,
    )
    if (
        set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization_receipt",
            "materiality_policy",
            "prior_v9_authority",
            "incident_authorities",
            "diagnosis",
            "ordinal_contract",
            "source_transition",
            "focused_test_evidence",
            "access_disclosure",
            "checkpoint_reuse",
            "non_economic_boundary",
            "replacement_authority",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-factor-covariance-ordinal-basis-alignment-v1"
        or amendment.get("classification")
        != (
            "non_economic_rule_restoring_zero_based_factor_covariance_"
            "ordinal_contract_alignment"
        )
        or sha256_bytes(canonical_json_bytes(amendment))
        != _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_SEMANTIC_SHA256
    ):
        raise Stage12FreezeError(
            "factor-covariance ordinal-basis amendment semantics differ"
        )

    authorization, authorization_record = bound_json(
        _as_mapping(
            amendment.get("authorization_receipt"),
            "factor-covariance ordinal-basis authorization is missing",
        ),
        expected_sha256=(
            _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AUTHORIZATION_SHA256
        ),
        expected_status=(
            "STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_" "AUTHORIZED"
        ),
        label="factor-covariance ordinal-basis authorization",
    )
    materiality, materiality_record = bound_json(
        _as_mapping(
            amendment.get("materiality_policy"),
            "thesis error-materiality policy is missing",
        ),
        expected_sha256=_EXPECTED_THESIS_ERROR_MATERIALITY_POLICY_SHA256,
        expected_status="THESIS_ERROR_MATERIALITY_POLICY_ACTIVE",
        label="thesis error-materiality policy",
    )
    incidents = _as_mapping(
        amendment.get("incident_authorities"),
        "factor-covariance incident authorities are missing",
    )
    blocker, blocker_record = bound_json(
        _as_mapping(
            incidents.get("blocker"),
            "factor-covariance blocker binding is missing",
        ),
        expected_sha256=(
            _EXPECTED_STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_BLOCKER_SHA256
        ),
        expected_status=(
            "STAGE13_FAILED_CLOSED_BEFORE_NORMALIZED_SOURCE_PUBLICATION_ON_"
            "FACTOR_COVARIANCE_ORDINAL_BASIS_MISMATCH"
        ),
        label="Stage 13 factor-covariance ordinal-basis blocker",
    )
    preflight, preflight_record = bound_json(
        _as_mapping(
            incidents.get("preflight"),
            "factor-covariance preflight binding is missing",
        ),
        expected_sha256=(
            _EXPECTED_STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_PREFLIGHT_SHA256
        ),
        expected_status=("STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_PREFLIGHT_COMPLETE"),
        label="Stage 13 factor-covariance ordinal-basis preflight",
    )
    failed_launch, failed_launch_record = bound_json(
        _as_mapping(
            incidents.get("failed_launch_memo"),
            "failed Stage 13 v9 launch memo binding is missing",
        ),
        expected_sha256=_EXPECTED_FAILED_STAGE13_V9_LAUNCH_MEMO_SHA256,
        expected_status=None,
        label="failed Stage 13 v9 launch memo",
        binding_status_required=False,
    )

    authorization_scope = _as_mapping(
        authorization.get("authorization"),
        "factor-covariance authorization scope is missing",
    )
    required_safety = _as_mapping(
        authorization.get("required_safety"),
        "factor-covariance authorization safety contract is missing",
    )
    diagnosis = _as_mapping(
        amendment.get("diagnosis"),
        "factor-covariance diagnosis is missing",
    )
    ordinal_contract = _as_mapping(
        amendment.get("ordinal_contract"),
        "factor-covariance ordinal contract is missing",
    )
    access = _as_mapping(
        amendment.get("access_disclosure"),
        "factor-covariance access disclosure is missing",
    )
    checkpoint = _as_mapping(
        amendment.get("checkpoint_reuse"),
        "factor-covariance checkpoint contract is missing",
    )
    boundary = _as_mapping(
        amendment.get("non_economic_boundary"),
        "factor-covariance non-economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "factor-covariance replacement authority is missing",
    )
    focused_tests = _as_mapping(
        amendment.get("focused_test_evidence"),
        "factor-covariance focused-test evidence is missing",
    )
    terminal = _as_mapping(
        failed_launch.get("terminal_update"),
        "failed Stage 13 v9 terminal update is missing",
    )
    if (
        authorization_scope.get("authorized") is not True
        or authorization_scope.get("routine_non_economic_repair_gate_reprompt_required")
        is not False
        or authorization_scope.get("stage13_execution_authorized_now") is not False
        or required_safety.get("covariance_daily_and_monthly_bits_must_not_change")
        is not True
        or required_safety.get("matrix_sort_reshape_and_factor_order_must_not_change")
        is not True
        or required_safety.get("risk_models_weights_trades_returns_and_pnl_may_change")
        is not False
        or required_safety.get("schemas_must_not_change") is not True
        or required_safety.get(
            "source_zero_based_ordinals_must_match_factor_name_indices"
        )
        is not True
        or required_safety.get("stage12_v10_independent_verification_required")
        is not True
        or diagnosis.get("months") != 72
        or diagnosis.get("covariance_rows") != 2592
        or diagnosis.get("source_ordinal_domain") != [0, 1, 2, 3, 4, 5]
        or diagnosis.get("incorrect_stage13_guard_domain") != [1, 2, 3, 4, 5, 6]
        or diagnosis.get("covariance_daily_bit_differences") != 0
        or diagnosis.get("covariance_monthly_bit_differences") != 0
        or diagnosis.get("monthly_scale_bit_differences_from_21_times_daily") != 0
        or diagnosis.get("factor_name_or_order_mismatches") != 0
        or diagnosis.get("nonfinite_cells") != 0
        or diagnosis.get("asymmetric_months") != 0
        or diagnosis.get("non_psd_months") != 0
        or float(diagnosis.get("minimum_monthly_eigenvalue", -1.0)) <= 0.0
        or ordinal_contract
        != {
            "accept_both_bases_permitted": False,
            "authoritative_basis": "zero_based_0_through_5",
            "renumber_stored_source_permitted": False,
            "stage11_materialization_verifier_requires_zero_based": True,
            "stage11_portfolio_backend_requires_zero_based": True,
            "stage13_compiler_must_require_zero_based": True,
            "stage13_runtime_must_require_zero_based": True,
            "stage8_generator_uses_zero_based_enumerate": True,
            "stage8_independent_verifier_requires_zero_based": True,
        }
        or preflight.get("rows_checked") != 2592
        or preflight.get("months_checked") != 72
        or preflight.get("zero_based_ordinal_mismatches") != 0
        or preflight.get("one_based_ordinal_mismatches") != 2592
        or preflight.get("covariance_daily_bit_differences") != 0
        or preflight.get("covariance_monthly_bit_differences") != 0
        or preflight.get("monthly_scale_bit_differences_from_21_times_daily") != 0
        or preflight.get("factor_name_or_order_mismatches") != 0
        or preflight.get("nonfinite_covariance_cells") != 0
        or preflight.get("asymmetric_months") != 0
        or preflight.get("non_psd_months") != 0
        or blocker.get("materiality_classification")
        != (
            "NON_ECONOMIC_STRUCTURAL_METADATA_CONTRACT_MISMATCH_NOT_"
            "NUMERICAL_BUFFER_ELIGIBLE"
        )
        or blocker.get("direct_bypass_permitted") is not False
        or blocker.get("covariance_daily_bit_differences") != 0
        or blocker.get("covariance_monthly_bit_differences") != 0
        or blocker.get("economic_checkpoint_count") != 0
        or blocker.get("normalized_source_published") is not False
        or blocker.get("input_registration_published") is not False
        or blocker.get("primary_result_published") is not False
        or blocker.get("verified_result_published") is not False
        or access.get("diagnostic_covariance_rows_read") != 2592
        or access.get("diagnostic_factor_covariance_parquet_files_opened") != 72
        or access.get("economic_checkpoint_files_opened") != 0
        or access.get("model_forecast_weight_trade_return_or_pnl_files_created") != 0
        or access.get("stage12_v10_parent_or_final_payload_files_opened_required") != 0
        or checkpoint.get("authenticated_parent_decision_checkpoint_pairs_to_rebind")
        != 431
        or checkpoint.get("economic_checkpoint_reuse_available") is not False
        or checkpoint.get("existing_decision_checkpoint_rebind_v2_unchanged")
        is not True
        or checkpoint.get("failed_v9_context_preserved_immutably") is not True
        or checkpoint.get("new_full_source_migration_authority_added") is not False
        or checkpoint.get("only_decision_checkpoint_to_rebuild")
        != "final_market_inputs/2025-12"
        or boundary
        != {
            "covariance_values_changed": False,
            "economic_month_checkpoints_before_failure": 0,
            "factor_names_or_order_changed": False,
            "matrix_sort_or_reshape_changed": False,
            "risk_formulas_models_weights_trades_returns_pnl_or_inference_changed": False,
            "schemas_changed": False,
        }
        or replacement.get("replacement_stage12_v10_identity_required") is not True
        or replacement.get("independent_stage12_verification_required") is not True
        or replacement.get("prior_v9_stage13_approval_valid_for_v10") is not False
        or replacement.get("fresh_exact_binding_stage13_receipt_required") is not True
        or replacement.get("standing_approval_permits_receipt_after_verification")
        is not True
        or replacement.get("additional_user_prompt_after_v10_verification_required")
        is not False
        or replacement.get("stage13_execution_authorized_now") is not False
        or focused_tests.get("tests_passed") != 60
        or failed_launch.get("stage") != 13
        or failed_launch.get("status")
        != (
            "FAILED_CLOSED_AFTER_790_SOURCE_CHECKPOINTS_BEFORE_NORMALIZED_"
            "SOURCE_PUBLICATION_ON_COVARIANCE_ORDINAL_BASIS_MISMATCH"
        )
        or terminal.get("source_checkpoint_count") != 790
        or terminal.get("source_checkpoint_validation_issues") != 0
        or terminal.get("affected_ordinal_rows") != 2592
        or terminal.get("covariance_numeric_value_differences") != 0
        or terminal.get("economic_month_checkpoints") != 0
        or terminal.get("normalized_source_published") is not False
        or terminal.get("input_registration_published") is not False
        or terminal.get("primary_result_published") is not False
        or terminal.get("verified_result_published") is not False
    ):
        raise Stage12FreezeError(
            "factor-covariance ordinal-basis alignment contract differs"
        )

    prior = _as_mapping(
        amendment.get("prior_v9_authority"),
        "prior v9 factor-covariance authority is missing",
    )
    expected_prior_files = {
        str(prior.get("primary_pointer_archive_path")): (
            "a9799dbf0e033c9ffb0acf771037bb31fd7bd5b50cb4434272d9cd9857755117"
        ),
        str(prior.get("verified_pointer_archive_path")): (
            "69bc5ad1514fefd5a21a1ab9b7318a26ecd89862f3f07e86f8d8ad166bea4ec0"
        ),
        str(prior.get("stage13_approval_archive_path")): (
            "ac4432190a5b0a33b8c892f14e13cde0f06bb97b5d4b49878b2e613061202408"
        ),
        str(prior.get("file_inventory_path")): (
            _PRIOR_V9_STAGE12_FILE_INVENTORY_SHA256
        ),
        str(prior.get("policy_path")): _EXPECTED_PRIOR_V9_REPLACEMENT_POLICY_SHA256,
    }
    if (
        prior.get("file_count") != 608
        or prior.get("project_tree_sha256")
        != "bf11ae118f4e39aa74834db70d8e157f0e797f22f7ec17c0199a9b309ce07ef1"
        or prior.get("freeze_id") != "stage12-freeze-bdf5a063d5fee375556d3d88"
        or prior.get("manifest_sha256")
        != "0242b9b86edb208950ce3e8ce522f7912672f9ee6ae57c514cbd474cbef4c503"
        or prior.get("primary_success_sha256")
        != "8d9aababe804c3a0170baf26c7add967e207c1f6700e8400d41849a93fde4e4c"
        or prior.get("verification_id") != "stage12-verify-f47beadcb21575341cfaefff"
        or prior.get("verification_report_sha256")
        != "82919322b647a92f5a146ef459ed4f82e83bbf4587c338254e615003c418c98e"
        or prior.get("verification_success_sha256")
        != "f043e8f57e86526de4f5e4b694587220644c0880846837767c4e45ed6d53b13f"
        or prior.get("stage12_binding_sha256")
        != "d7f20e9355ee8ba0f3be152dc8534ae5172a7d40a465f58d0519c2019e5604b3"
        or prior.get("stage13_approval_payload_sha256")
        != "80b8d5021cac217fd5641d7afd67af3be0637f3b6d85d74faf3d81df42fa8cb8"
    ):
        raise Stage12FreezeError("prior v9 factor-covariance authority differs")
    for relative_path, expected_sha in expected_prior_files.items():
        try:
            selected = confined_relative_path(
                project_root,
                relative_path,
                file_required=True,
                label="prior v9 factor-covariance authority",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("prior v9 authority path is unsafe") from exc
        if record.sha256 != expected_sha:
            raise Stage12FreezeError("prior v9 authority identity differs")

    expected_members = [
        {
            "path": "src/portfolio_bot/final_test/final_input_construction.py",
            "pre_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_COMPILER_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_COMPILER_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/production_runtime.py",
            "pre_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_RUNTIME_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_RUNTIME_POST_SHA256,
        },
    ]
    transition = _as_mapping(
        amendment.get("source_transition"),
        "factor-covariance source transition is missing",
    )
    if (
        transition.get("exact_changed_runtime_member_count") != 2
        or transition.get("members") != expected_members
    ):
        raise Stage12FreezeError("factor-covariance source transition differs")
    for member in expected_members:
        try:
            selected = confined_relative_path(
                project_root,
                member["path"],
                file_required=True,
                label="factor-covariance runtime member",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "factor-covariance runtime path is unsafe"
            ) from exc
        expected_current_sha = (
            _EXPECTED_EQUAL_WEIGHT_RUNTIME_POST_SHA256
            if member["path"] == "src/portfolio_bot/final_test/production_runtime.py"
            else member["post_amendment_sha256"]
        )
        if not _is_consolidated_v12(policy) and record.sha256 != expected_current_sha:
            raise Stage12FreezeError(
                "factor-covariance runtime source identity differs"
            )

    expected_test_files = [
        {
            "path": "tests/unit/test_final_input_construction.py",
            "pre_amendment_sha256": (
                "d34cacd45cf2b8bd520d1a656dbd2a6ff4ff510113beb298cadb756fc0f151ea"
            ),
            "post_amendment_sha256": (
                "a3fc7548a75c27a76a3b68bc68cfa8ab338ffb14ae47bfeaad58554bb7f5b4cf"
            ),
        },
        {
            "path": "tests/unit/test_final_source_builder.py",
            "pre_amendment_sha256": (
                "828d86021a4b511c1a892817ef56dd56f91b0b41f0cf558ab2cd5ca757755824"
            ),
            "post_amendment_sha256": (
                "49084d2ce45ca83ac7782140f1219ce33a3f161b3048c9a478c2372e6fecd9e1"
            ),
        },
        {
            "path": "tests/unit/test_final_test_production_runtime.py",
            "pre_amendment_sha256": (
                "559ca9cd58cb45cb1f5f033defad59b97577e035abad68909fe1b9b5a2303904"
            ),
            "post_amendment_sha256": (
                "3458d95cf8fae3bbb9401657dbcbcdc30894f348113c629a9db9163d2880b016"
            ),
        },
    ]
    if focused_tests.get("test_files") != expected_test_files:
        raise Stage12FreezeError("factor-covariance focused-test identities differ")
    for test_file in expected_test_files:
        try:
            selected = confined_relative_path(
                project_root,
                test_file["path"],
                file_required=True,
                label="factor-covariance focused-test member",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "factor-covariance focused-test path is unsafe"
            ) from exc
        if (
            not _is_consolidated_v12(policy)
            and record.sha256 != test_file["post_amendment_sha256"]
        ):
            raise Stage12FreezeError(
                "factor-covariance focused-test source identity differs"
            )

    return amendment, {
        "relative_path": (
            POST_UNBLIND_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT.relative_to(
                project_root
            ).as_posix()
        ),
        "sha256": amendment_record.sha256,
        "semantic_sha256": (
            _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_SEMANTIC_SHA256
        ),
        "size_bytes": amendment_record.size_bytes,
        "authorization": {
            "sha256": authorization_record.sha256,
            "size_bytes": authorization_record.size_bytes,
        },
        "materiality_policy": {
            "sha256": materiality_record.sha256,
            "size_bytes": materiality_record.size_bytes,
        },
        "blocker": {
            "sha256": blocker_record.sha256,
            "size_bytes": blocker_record.size_bytes,
        },
        "preflight": {
            "sha256": preflight_record.sha256,
            "size_bytes": preflight_record.size_bytes,
        },
        "failed_launch_memo": {
            "sha256": failed_launch_record.sha256,
            "size_bytes": failed_launch_record.size_bytes,
        },
    }


def _validate_post_unblind_equal_weight_missing_cost_gate_amendment(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Authenticate the rule-restoring equal-weight execution gate correction."""

    binding = _as_mapping(
        policy.get("post_unblind_equal_weight_missing_cost_gate_amendment"),
        "equal-weight missing-cost gate amendment binding is missing",
    )
    expected_binding = {
        "path": (
            "configs/phase_06_final_test/"
            "stage_12_post_unblind_equal_weight_missing_cost_gate_amendment_v1.json"
        ),
        "sha256": _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SHA256,
    }
    _require_exact_mapping(
        binding,
        expected_binding,
        label="equal-weight missing-cost gate amendment binding",
    )

    def bound_json(
        raw: Mapping[str, Any],
        *,
        expected_sha256: str,
        expected_status: str,
        label: str,
        binding_status_required: bool = True,
    ) -> tuple[Mapping[str, Any], StableFile]:
        try:
            path = confined_relative_path(
                project_root,
                raw.get("path"),
                file_required=True,
                label=label,
            )
            value, record = load_json_object(path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if (
            raw.get("sha256") != expected_sha256
            or record.sha256 != expected_sha256
            or (binding_status_required and raw.get("status") != expected_status)
            or value.get("status") != expected_status
        ):
            raise Stage12FreezeError(f"{label} identity differs")
        return value, record

    amendment, amendment_record = bound_json(
        binding,
        expected_sha256=_EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SHA256,
        expected_status="STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED",
        label="equal-weight missing-cost gate amendment",
        binding_status_required=False,
    )
    if (
        set(amendment)
        != {
            "schema_version",
            "stage",
            "amendment_id",
            "status",
            "classification",
            "authorization_receipt",
            "materiality_policy",
            "prior_v10_authority",
            "incident_authorities",
            "diagnosis",
            "execution_gate_contract",
            "source_transition",
            "focused_test_evidence",
            "access_disclosure",
            "checkpoint_reuse",
            "economic_boundary",
            "replacement_authority",
        }
        or amendment.get("schema_version") != 1
        or amendment.get("stage") != 12
        or amendment.get("amendment_id")
        != "stage12-post-unblind-equal-weight-missing-cost-gate-v1"
        or amendment.get("classification")
        != "rule_restoring_economic_benchmark_execution_gate_correction"
        or sha256_bytes(canonical_json_bytes(amendment))
        != _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SEMANTIC_SHA256
    ):
        raise Stage12FreezeError(
            "equal-weight missing-cost gate amendment semantics differ"
        )

    authorization, authorization_record = bound_json(
        _as_mapping(
            amendment.get("authorization_receipt"),
            "equal-weight missing-cost gate authorization is missing",
        ),
        expected_sha256=_EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AUTHORIZATION_SHA256,
        expected_status="STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED",
        label="equal-weight missing-cost gate authorization",
    )
    materiality, materiality_record = bound_json(
        _as_mapping(
            amendment.get("materiality_policy"),
            "thesis error-materiality policy is missing",
        ),
        expected_sha256=_EXPECTED_THESIS_ERROR_MATERIALITY_POLICY_SHA256,
        expected_status="THESIS_ERROR_MATERIALITY_POLICY_ACTIVE",
        label="thesis error-materiality policy",
    )
    incidents = _as_mapping(
        amendment.get("incident_authorities"),
        "equal-weight missing-cost incident authorities are missing",
    )
    failure, failure_record = bound_json(
        _as_mapping(
            incidents.get("resume3_failure"),
            "Resume 3 failure binding is missing",
        ),
        expected_sha256=_EXPECTED_STAGE13_RESUME3_FAILURE_SHA256,
        expected_status="STAGE13_RESUME3_FAILED_CLOSED_BEFORE_FIRST_ECONOMIC_CHECKPOINT",
        label="Stage 13 Resume 3 failure",
    )
    preflight, preflight_record = bound_json(
        _as_mapping(
            incidents.get("preflight"),
            "equal-weight missing-cost preflight binding is missing",
        ),
        expected_sha256=_EXPECTED_STAGE13_EQUAL_WEIGHT_MISSING_COST_PREFLIGHT_SHA256,
        expected_status="STAGE13_EQUAL_WEIGHT_MISSING_COST_GATE_PREFLIGHT_COMPLETE",
        label="Stage 13 equal-weight missing-cost preflight",
    )

    authorization_scope = _as_mapping(
        authorization.get("authorization"),
        "equal-weight authorization scope is missing",
    )
    materiality_scope = _as_mapping(
        authorization.get("controlling_materiality_policy"),
        "equal-weight materiality scope is missing",
    )
    authorized_runtime = _as_mapping(
        authorization.get("authorized_runtime_semantics"),
        "equal-weight authorized runtime semantics are missing",
    )
    diagnosis = _as_mapping(
        amendment.get("diagnosis"),
        "equal-weight missing-cost diagnosis is missing",
    )
    execution_gate = _as_mapping(
        amendment.get("execution_gate_contract"),
        "equal-weight execution-gate contract is missing",
    )
    access = _as_mapping(
        amendment.get("access_disclosure"),
        "equal-weight access disclosure is missing",
    )
    checkpoint = _as_mapping(
        amendment.get("checkpoint_reuse"),
        "equal-weight checkpoint contract is missing",
    )
    economic = _as_mapping(
        amendment.get("economic_boundary"),
        "equal-weight economic boundary is missing",
    )
    replacement = _as_mapping(
        amendment.get("replacement_authority"),
        "equal-weight replacement authority is missing",
    )
    failure_event = _as_mapping(
        failure.get("failure"),
        "Resume 3 failure event is missing",
    )
    preflight_domain = _as_mapping(
        preflight.get("empirical_domain"),
        "equal-weight preflight empirical domain is missing",
    )
    preflight_rule = _as_mapping(
        preflight.get("required_rule"),
        "equal-weight preflight rule is missing",
    )
    preflight_forward = _as_mapping(
        preflight.get("empirical_forward_compatibility_audit"),
        "equal-weight forward compatibility audit is missing",
    )
    if (
        authorization_scope.get("authorized") is not True
        or authorization_scope.get(
            "additional_user_prompt_after_v11_verification_required"
        )
        is not False
        or authorization_scope.get("stage13_execution_authorized_now") is not False
        or materiality_scope.get(
            "valid_error_affecting_thesis_economic_meaning_must_be_fixed"
        )
        is not True
        or materiality_scope.get("negligibility_exception_used") is not False
        or authorized_runtime.get(
            "invalid_execution_or_cost_input_remains_zero_capacity_regardless_of_entry_gate"
        )
        is not True
        or authorized_runtime.get(
            "new_or_increased_long_requires_complete_current_authenticated_risk_inputs"
        )
        is not True
        or authorized_runtime.get(
            "nonzero_trade_with_invalid_execution_or_cost_input_fails_closed"
        )
        is not True
        or authorized_runtime.get(
            "frozen_inherited_exact_hold_with_missing_current_beta_uses_last_authenticated_inherited_beta"
        )
        is not True
        or authorized_runtime.get(
            "inherited_beta_fallback_may_not_authorize_new_or_increased_risk"
        )
        is not True
        or authorized_runtime.get(
            "nonzero_target_without_current_or_inherited_authenticated_beta_fails_closed"
        )
        is not True
        or authorized_runtime.get("other_five_strategy_rules_unchanged") is not True
        or diagnosis.get("affected_long_eligible_missing_adv_rows") != 25
        or diagnosis.get("affected_distinct_permnos") != 23
        or diagnosis.get("affected_holding_months") != 19
        or diagnosis.get("january_2020_affected_rows") != 2
        or diagnosis.get("all_affected_new_long_trade_allowed_false") is not True
        or diagnosis.get("optimized_strategy_problem_builder_already_enforces_gate")
        is not True
        or diagnosis.get("equal_weight_branch_omitted_gate") is not True
        or diagnosis.get("common_cost_engine_behavior_correct") is not True
        or diagnosis.get("same_v10_restart_would_repeat_failure") is not True
        or diagnosis.get(
            "equal_weight_inherited_current_invalid_execution_or_cost_transitions"
        )
        != 24
        or diagnosis.get("missing_adv_with_current_risk_present") != 22
        or diagnosis.get("missing_entry_close_with_current_risk_absent") != 2
        or diagnosis.get("missing_entry_close_rows_requiring_inherited_beta_fallback")
        != [
            {"holding_month": "2020-05", "permno": 18662},
            {"holding_month": "2022-03", "permno": 12799},
        ]
        or preflight_domain.get("long_eligible_missing_adv_rows") != 25
        or preflight_domain.get("long_eligible_missing_adv_distinct_permnos") != 23
        or preflight_domain.get("affected_holding_months") != 19
        or preflight_domain.get("all_new_long_trade_allowed_false") is not True
        or preflight_rule.get(
            "new_or_increased_position_requires_long_eligible_gate_open_and_valid_execution_cost_inputs"
        )
        is not True
        or preflight_rule.get(
            "new_or_increased_position_requires_complete_current_authenticated_risk_inputs"
        )
        is not True
        or preflight_rule.get(
            "invalid_execution_or_cost_input_is_never_executable_regardless_of_entry_gate"
        )
        is not True
        or preflight_rule.get(
            "nonzero_trade_with_invalid_execution_or_cost_input_fails_closed"
        )
        is not True
        or preflight_rule.get(
            "frozen_inherited_exact_hold_with_missing_current_beta_uses_last_authenticated_inherited_beta"
        )
        is not True
        or preflight_rule.get(
            "inherited_beta_fallback_may_not_authorize_new_or_increased_risk"
        )
        is not True
        or preflight_rule.get(
            "nonzero_target_without_current_or_inherited_authenticated_beta_fails_closed"
        )
        is not True
        or preflight_forward.get(
            "equal_weight_inherited_current_invalid_execution_or_cost_transitions"
        )
        != 24
        or preflight_forward.get("missing_adv_with_current_risk_present") != 22
        or preflight_forward.get("missing_entry_close_with_current_risk_absent") != 2
        or preflight_forward.get("inherited_beta_fallback_required") is not True
        or preflight_forward.get(
            "stage11_zero_capacity_applies_independently_of_entry_gate"
        )
        is not True
        or execution_gate.get(
            "new_or_increased_long_requires_long_eligible_open_trade_gate_and_valid_execution_cost_inputs"
        )
        is not True
        or execution_gate.get(
            "new_or_increased_long_requires_complete_finite_current_factor_betas_and_positive_finite_specific_variance"
        )
        is not True
        or execution_gate.get("required_exit_is_excluded_from_executable_long")
        is not True
        or execution_gate.get(
            "missing_execution_or_cost_inherited_long_exact_hold_required"
        )
        is not True
        or execution_gate.get("frozen_inherited_trade_exactly_zero") is not True
        or execution_gate.get(
            "frozen_hold_and_zero_trade_use_exact_equality_not_tolerance"
        )
        is not True
        or execution_gate.get("residual_gross_equals_one_minus_frozen_long_gross")
        is not True
        or execution_gate.get(
            "residual_gross_equal_across_executable_long_eligible_rows"
        )
        is not True
        or execution_gate.get("frozen_gross_above_one_fails_closed") is not True
        or execution_gate.get(
            "invalid_execution_or_cost_input_is_zero_capacity_regardless_of_entry_gate"
        )
        is not True
        or execution_gate.get(
            "nonzero_trade_with_invalid_execution_or_cost_input_fails_closed"
        )
        is not True
        or execution_gate.get(
            "frozen_inherited_exact_hold_with_missing_current_beta_uses_last_authenticated_inherited_beta"
        )
        is not True
        or execution_gate.get(
            "inherited_beta_fallback_may_not_authorize_new_or_increased_risk"
        )
        is not True
        or execution_gate.get(
            "nonzero_target_without_current_or_inherited_authenticated_beta_fails_closed"
        )
        is not True
        or execution_gate.get("positive_adv_imputation_permitted") is not False
        or execution_gate.get("authenticated_row_drop_permitted") is not False
        or execution_gate.get("cost_guard_bypass_permitted") is not False
        or execution_gate.get(
            "blocked_flat_rows_use_descriptive_blocked_candidate_class"
        )
        is not True
        or execution_gate.get("candidate_counts_are_derived_from_executable_rows")
        is not True
        or execution_gate.get(
            "new_risk_permitted_is_derived_from_executable_count_and_positive_residual"
        )
        is not True
        or execution_gate.get("target_gross") != 1.0
        or failure_event.get("strategy_id") != "equal_weight_long_only_scoring_universe"
        or failure_event.get("holding_month") != "2020-01"
        or failure_event.get("exception_type") != "CostInputError"
        or access.get("accepted_trade_cost_return_pnl_or_performance_files_opened") != 0
        or access.get("economic_checkpoint_files_accepted") != 0
        or access.get("stage12_v11_parent_or_final_payload_files_opened_required") != 0
        or checkpoint.get("resume3_economic_checkpoint_count") != 0
        or checkpoint.get("economic_checkpoint_reuse_available") is not False
        or checkpoint.get("v10_model_state_direct_reuse_permitted") is not False
        or checkpoint.get("fresh_v11_full_and_reduced_model_fits_required") != 6
        or checkpoint.get("restart_boundary")
        != "primary_full_2020-01_ordinal_1_predecessor_null"
        or economic.get("equal_weight_benchmark_outputs_may_change") is not True
        or economic.get("change_required_to_restore_prespecified_common_execution_gate")
        is not True
        or economic.get("models_scores_ranks_and_other_five_strategy_rules_changed")
        is not False
        or economic.get("common_transaction_cost_model_changed") is not False
        or economic.get("cost_input_transport_changed") is not False
        or economic.get("stage1_through_stage11_retraining_required") is not False
        or economic.get("outcome_months_accepted_before_failure") != 0
        or replacement.get("replacement_stage12_v11_identity_required") is not True
        or replacement.get("independent_stage12_verification_required") is not True
        or replacement.get("prior_v10_stage13_approval_valid_for_v11") is not False
        or replacement.get("fresh_exact_binding_stage13_approval_required") is not True
        or replacement.get("standing_approval_permits_receipt_after_verification")
        is not True
        or replacement.get("additional_user_prompt_after_v11_verification_required")
        is not False
        or replacement.get("stage13_execution_authorized_now") is not False
    ):
        raise Stage12FreezeError(
            "equal-weight missing-cost gate amendment contract differs"
        )

    prior = _as_mapping(
        amendment.get("prior_v10_authority"),
        "prior v10 equal-weight authority is missing",
    )
    if (
        prior.get("policy_sha256") != _EXPECTED_PRIOR_V10_REPLACEMENT_POLICY_SHA256
        or prior.get("file_inventory_sha256")
        != _PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256
        or prior.get("file_count") != 610
        or prior.get("project_tree_sha256")
        != "77a08b63b34e93a9750e0e820b6329b3817cbf50f5970999d0087a593cd6c032"
        or prior.get("freeze_id") != "stage12-freeze-4159858fb5e321c1ec45c3fd"
        or prior.get("stage12_binding_sha256")
        != "be92d206989305525d946ffb6093cc160bb5bdd88da671ddc00ef2338f586a03"
    ):
        raise Stage12FreezeError("prior v10 equal-weight authority differs")
    for path_key, sha_key in (
        ("policy_path", "policy_sha256"),
        ("file_inventory_path", "file_inventory_sha256"),
    ):
        try:
            selected = confined_relative_path(
                project_root,
                prior.get(path_key),
                file_required=True,
                label="prior v10 equal-weight authority",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("prior v10 authority path is unsafe") from exc
        if record.sha256 != prior.get(sha_key):
            raise Stage12FreezeError("prior v10 authority identity differs")

    transition = _as_mapping(
        amendment.get("source_transition"),
        "equal-weight source transition is missing",
    )
    expected_members = [
        {
            "path": "src/portfolio_bot/final_test/economic_kernel.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_KERNEL_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_KERNEL_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/publication.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_PUBLICATION_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_PUBLICATION_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/production_runtime.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_RUNTIME_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_RUNTIME_POST_SHA256,
        },
    ]
    if (
        transition.get("exact_changed_runtime_member_count") != 3
        or transition.get("members") != expected_members
    ):
        raise Stage12FreezeError("equal-weight source transition differs")
    focused_tests = _as_mapping(
        amendment.get("focused_test_evidence"),
        "equal-weight focused-test evidence is missing",
    )
    expected_test_files = [
        {
            "path": "tests/unit/test_final_test_economic_kernel.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_TEST_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_TEST_POST_SHA256,
        },
        {
            "path": "tests/unit/test_final_test_publication.py",
            "pre_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_PUBLICATION_TEST_PRE_SHA256
            ),
            "post_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_PUBLICATION_TEST_POST_SHA256
            ),
        },
    ]
    if (
        focused_tests.get("tests_passed") != 58
        or focused_tests.get("suite_counts")
        != {"economic_kernel": 29, "publication": 14, "production_runtime": 15}
        or focused_tests.get("test_files") != expected_test_files
        or focused_tests.get("unchanged_support_test_files")
        != [
            {
                "path": "tests/unit/test_final_test_production_runtime.py",
                "sha256": (
                    "3458d95cf8fae3bbb9401657dbcbcdc30894f348113c629a9db9163d2880b016"
                ),
            }
        ]
    ):
        raise Stage12FreezeError("equal-weight focused-test identities differ")
    for member in expected_members + expected_test_files:
        try:
            selected = confined_relative_path(
                project_root,
                member["path"],
                file_required=True,
                label="equal-weight amended source member",
            )
            record = stable_file(selected, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                "equal-weight amended source path is unsafe"
            ) from exc
        if (
            not _is_consolidated_v12(policy)
            and record.sha256 != member["post_amendment_sha256"]
        ):
            raise Stage12FreezeError("equal-weight amended source identity differs")
    try:
        support_test_path = confined_relative_path(
            project_root,
            "tests/unit/test_final_test_production_runtime.py",
            file_required=True,
            label="equal-weight unchanged support-test member",
        )
        support_test_record = stable_file(
            support_test_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("equal-weight support-test path is unsafe") from exc
    if not _is_consolidated_v12(policy) and support_test_record.sha256 != (
        "3458d95cf8fae3bbb9401657dbcbcdc30894f348113c629a9db9163d2880b016"
    ):
        raise Stage12FreezeError("equal-weight support-test identity differs")

    return amendment, {
        "relative_path": str(binding["path"]),
        "sha256": amendment_record.sha256,
        "semantic_sha256": (
            _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SEMANTIC_SHA256
        ),
        "size_bytes": amendment_record.size_bytes,
        "authorization": {
            "sha256": authorization_record.sha256,
            "size_bytes": authorization_record.size_bytes,
        },
        "materiality_policy": {
            "sha256": materiality_record.sha256,
            "size_bytes": materiality_record.size_bytes,
        },
        "resume3_failure": {
            "sha256": failure_record.sha256,
            "size_bytes": failure_record.size_bytes,
        },
        "preflight": {
            "sha256": preflight_record.sha256,
            "size_bytes": preflight_record.size_bytes,
        },
    }


def _validate_stage13_completion_consolidation(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate the v12 consolidation and its substantive 2021 treatment."""

    consolidation = _as_mapping(
        policy.get("consolidated_stage13_completion_governance"),
        "Stage 12 v12 completion consolidation is missing",
    )
    required_keys = {
        "status",
        "classification",
        "active_prior_authority",
        "unpublished_v11",
        "terminal_event_treatment",
        "final_input_registry_contract",
        "report_template_contract",
        "result_assembly_contract",
        "implementation_members",
        "focused_test_members",
        "access_and_approval_boundary",
        "inventory_delta_authorization",
    }
    if set(consolidation) != required_keys:
        raise Stage12FreezeError("Stage 12 v12 consolidation keys differ")
    if consolidation.get(
        "status"
    ) != "STAGE12_V12_CONSOLIDATED_STAGE13_COMPLETION_GOVERNANCE_FROZEN" or consolidation.get(
        "classification"
    ) != (
        "mixed_rule_restoring_technical_and_substantive_economic_"
        "post_unblind_replacement"
    ):
        raise Stage12FreezeError("Stage 12 v12 consolidation identity differs")
    _require_exact_mapping(
        _as_mapping(
            consolidation.get("active_prior_authority"),
            "active prior Stage 12 authority is missing",
        ),
        {
            "version": 10,
            "policy_path": "configs/phase_06_final_test/stage_12_freeze_policy_v10.json",
            "policy_sha256": _EXPECTED_PRIOR_V10_REPLACEMENT_POLICY_SHA256,
            "freeze_id": "stage12-freeze-4159858fb5e321c1ec45c3fd",
            "stage12_binding_sha256": (
                "be92d206989305525d946ffb6093cc160bb5bdd88da671ddc00ef2338f586a03"
            ),
        },
        label="active prior Stage 12 authority",
    )
    _require_exact_mapping(
        _as_mapping(
            consolidation.get("unpublished_v11"),
            "unpublished Stage 12 v11 disposition is missing",
        ),
        {
            "policy_path": "configs/phase_06_final_test/stage_12_freeze_policy_v11.json",
            "policy_sha256": _EXPECTED_UNPUBLISHED_V11_POLICY_SHA256,
            "publication_status": "never_published",
            "superseded_by": "stage12_v12",
            "may_be_treated_as_prior_authority": False,
        },
        label="unpublished Stage 12 v11 disposition",
    )
    terminal = _as_mapping(
        consolidation.get("terminal_event_treatment"),
        "substantive terminal-event treatment is missing",
    )
    terminal_policy_binding = _as_mapping(
        policy.get("stage13_public_interest_last_tradable_close_liquidation"),
        "Stage 13 terminal-treatment policy binding is missing",
    )
    terminal_policy_sha256 = str(terminal_policy_binding.get("sha256"))
    authorization_basis = _as_mapping(
        terminal.get("authorization_basis"),
        "substantive terminal-event authorization basis is missing",
    )
    standing_instruction = "WHY DID WE STOP PLEASE I WANT TO FINISH STAGE 13"
    standing_instruction_sha256 = (
        "13f24431ba741ce24158877302c803cb18a7f8a4ae435e67376594edeef05681"
    )
    if (
        terminal.get("classification")
        != "substantive_economic_post_unblind_data_gap_treatment"
        or terminal.get("policy_path")
        != (
            "configs/phase_06_final_test/"
            "stage_13_public_interest_last_tradable_close_liquidation_v1.json"
        )
        or terminal.get("policy_sha256") != terminal_policy_sha256
        or terminal.get("exact_expected_event_count") != 2
        or terminal.get("exact_permnos") != [85421, 88888]
        or terminal.get("primary_settlement_value_multiplier") != 1.0
        or terminal.get("vendor_missing_delisting_return_relabelled_as_zero")
        is not False
        or terminal.get("source_missing_return_fields_preserved") is not True
        or terminal.get("primary_results_may_change") is not True
        or terminal.get("required_counterfactual_value_multipliers") != [0.7, 0.45, 0.0]
        or terminal.get("required_reporting_only_scenario_count") != 4
        or terminal.get("execution_cost_stress_is_separate_not_cartesian") is not True
        or terminal.get("within_holding_month_cash_carry_required") is not True
        or terminal.get("primary_estimates_modified_by_sensitivity") is not False
        or authorization_basis
        != {
            "standing_user_instruction": standing_instruction,
            "standing_user_instruction_sha256": standing_instruction_sha256,
            "fresh_v12_binding_specific_stage13_approval_must_carry_this_basis_before_execution": True,
        }
        or sha256_bytes(standing_instruction.encode("utf-8"))
        != standing_instruction_sha256
    ):
        raise Stage12FreezeError(
            "substantive terminal-event treatment contract differs"
        )
    try:
        terminal_path = confined_relative_path(
            project_root,
            terminal["policy_path"],
            file_required=True,
            label="substantive terminal-event treatment policy",
        )
        terminal_policy, terminal_record = load_json_object(
            terminal_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "substantive terminal-event treatment policy path is unsafe"
        ) from exc
    terminal_scope = _as_mapping(
        terminal_policy.get("scope"), "terminal-event treatment scope is missing"
    )
    terminal_treatment = _as_mapping(
        terminal_policy.get("treatment"),
        "terminal-event accounting treatment is missing",
    )
    terminal_sensitivity = _as_mapping(
        terminal_policy.get("sensitivity"),
        "terminal-event sensitivity contract is missing",
    )
    terminal_governance = _as_mapping(
        terminal_policy.get("governance"),
        "terminal-event governance is missing",
    )
    if (
        terminal_record.sha256 != terminal_policy_sha256
        or terminal_policy.get("schema_version") != 1
        or terminal_policy.get("policy_id")
        != "stage13-public-interest-last-tradable-close-liquidation-v1"
        or terminal_policy.get("status") != "post_unblind_economic_data_gap_treatment"
        or terminal_scope.get("expected_match_count") != 2
        or [item.get("permno") for item in terminal_scope.get("expected_matches", [])]
        != [85421, 88888]
        or terminal_treatment.get("ordinary_final_session_return_authorized")
        is not True
        or terminal_treatment.get("vendor_post_delisting_return_relabelled_as_zero")
        is not False
        or terminal_treatment.get(
            "vendor_missing_return_fields_preserved_in_source_evidence"
        )
        is not True
        or terminal_treatment.get("settlement_kind") != "cash"
        or terminal_treatment.get("settlement_value_multiplier") != 1.0
        or terminal_treatment.get("terminal_total_return_already_applied") is not True
        or terminal_treatment.get("apply_order")
        != "observed_final_session_return_then_single_cash_retirement"
        or terminal_sensitivity.get("status")
        != "mandatory_reporting_only_one_at_a_time_counterfactuals"
        or terminal_sensitivity.get("counterfactual_post_delisting_value_multipliers")
        != [0.7, 0.45, 0.0]
        or terminal_sensitivity.get("counterfactual_multiplier_interpretation")
        != "replace_multiplier_one_final_close_cash_settlement_after_observed_ordinary_final_session_return"
        or terminal_sensitivity.get(
            "counterfactual_multiplier_applies_to_signed_post_return_pre_settlement_position_value"
        )
        is not True
        or terminal_sensitivity.get(
            "counterfactual_multiplier_is_not_an_imputed_vendor_delisting_return"
        )
        is not True
        or terminal_sensitivity.get("vendor_missing_return_fields_remain_missing")
        is not True
        or terminal_sensitivity.get("terminal_execution_cost_stress")
        != "maximum_of_authenticated_event_day_quoted_full_spread_and_frozen_full_spread_20d"
        or terminal_sensitivity.get("event_day_quoted_full_spread_formula")
        != "(ask_usd_per_share-bid_usd_per_share)/((ask_usd_per_share+bid_usd_per_share)/2)"
        or terminal_sensitivity.get("execution_cost_charge")
        != "one_full_selected_spread_times_absolute_post_return_pre_settlement_position_value"
        or terminal_sensitivity.get(
            "execution_cost_scenario_is_separate_one_at_a_time_not_cartesian"
        )
        is not True
        or terminal_sensitivity.get("within_holding_month_cash_carry")
        != "counterfactual_cash_difference_compounds_by_observed_remaining_daily_risk_free_growth_after_event_session"
        or terminal_sensitivity.get(
            "subsequent_primary_monthly_percentage_returns_unchanged"
        )
        is not True
        or terminal_sensitivity.get("primary_estimates_modified") is not False
        or terminal_sensitivity.get(
            "model_fitting_scoring_selection_or_allocation_rerun"
        )
        is not False
        or terminal_sensitivity.get("strategy_ranking_rule")
        != "sharpe_descending_then_net_cagr_descending_then_strategy_id_ascending"
        or terminal_sensitivity.get("final_thesis_conclusion_status")
        != "pending_stage14_inference_and_stage15_synthesis"
        or terminal_sensitivity.get("required_reported_effects")
        != [
            "event_day_portfolio_weight",
            "holding_month_return",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "strategy_ranking",
            "thesis_conclusion",
        ]
        or terminal_sensitivity.get("counterfactuals_are_primary_estimates")
        is not False
        or terminal_sensitivity.get("scenario_results_may_be_promoted") is not False
        or terminal_governance.get("post_unblind_substantive_amendment") is not True
        or terminal_governance.get("replacement_stage12_freeze_required") is not True
        or terminal_governance.get("independent_stage12_verification_required")
        is not True
        or terminal_governance.get("fresh_binding_specific_stage13_approval_required")
        is not True
    ):
        raise Stage12FreezeError(
            "substantive terminal-event treatment policy semantics differ"
        )
    _require_exact_mapping(
        _as_mapping(
            consolidation.get("final_input_registry_contract"),
            "Stage 13 final-input registry contract is missing",
        ),
        {
            "path": "configs/phase_06_final_test/stage_13_input_registry_v1.json",
            "registry_id": "stage13-final-input-registry-v1",
            "schema_version": 1,
            "template_count": 32,
            "total_arrow_column_count": 1210,
            "classification": (
                "authorized_terminal_sensitivity_evidence_transport_schema"
            ),
        },
        label="Stage 13 final-input registry contract",
    )
    _require_exact_mapping(
        _as_mapping(
            consolidation.get("report_template_contract"),
            "Stage 13 report-template contract is missing",
        ),
        {
            "historical_stage13_paths": [
                "configs/phase_06_final_test/stage_13_report_template_v1.md"
            ],
            "active_stage13_path": (
                "configs/phase_06_final_test/stage_13_report_template_v2.md"
            ),
            "active_stage14_path": (
                "configs/phase_06_final_test/stage_14_report_template_v1.md"
            ),
            "active_stage13_template_version": 2,
            "historical_templates_may_be_rewritten": False,
        },
        label="Stage 13 report-template contract",
    )
    _require_exact_mapping(
        _as_mapping(
            consolidation.get("result_assembly_contract"),
            "Stage 13 result-assembly contract is missing",
        ),
        {
            "assembler_id": "stage13-empirical-result-assembler-v3",
            "implementation_path": (
                "src/portfolio_bot/final_test/production_results.py"
            ),
            "economic_document_count": 13,
            "terminal_sensitivity_scenario_count": 4,
            "primary_estimates_modified": False,
            "model_fitting_scoring_selection_or_allocation_rerun": False,
        },
        label="Stage 13 result-assembly contract",
    )
    implementation_members = consolidation.get("implementation_members")
    focused_test_members = consolidation.get("focused_test_members")
    if (
        not isinstance(implementation_members, list)
        or not implementation_members
        or not isinstance(focused_test_members, list)
        or not focused_test_members
    ):
        raise Stage12FreezeError(
            "Stage 12 v12 consolidated member inventory is missing"
        )
    implementation_paths = {
        str(item.get("path"))
        for item in implementation_members
        if isinstance(item, Mapping)
    }
    expected_implementation_classifications = {
        "configs/phase_06_final_test/stage_13_input_registry_v1.json": (
            "authorized_terminal_sensitivity_evidence_transport_schema"
        ),
        "configs/phase_06_final_test/stage_13_public_interest_last_tradable_close_liquidation_v1.json": (
            "substantive_economic_terminal_treatment_and_reporting_only_sensitivity_policy"
        ),
        "configs/phase_06_final_test/stage_13_report_template_v2.md": (
            "active_stage13_terminal_sensitivity_reporting_contract"
        ),
        "src/portfolio_bot/allocation/solver.py": (
            "rule_preserving_legacy_weight_api_compatibility_and_unchanged_"
            "residue_evidence"
        ),
        "src/portfolio_bot/final_test/economic_kernel.py": (
            "rule_restoring_equal_weight_and_terminal_sensitivity_observation_capture"
        ),
        "src/portfolio_bot/final_test/final_input_construction.py": (
            "non_economic_v10_to_v12_side_by_side_input_namespace_transition"
        ),
        "src/portfolio_bot/final_test/final_raw_source_publication.py": (
            "non_economic_exact_predecessor_source_pointer_transition_and_v2_publication"
        ),
        "src/portfolio_bot/final_test/final_source_builder.py": (
            "terminal_policy_formula_and_quote_evidence_binding"
        ),
        "src/portfolio_bot/final_test/final_source_replay.py": (
            "substantive_terminal_treatment_sensitivity_and_2025_boundary_source_replay"
        ),
        "src/portfolio_bot/final_test/input_materialization.py": (
            "terminal_sensitivity_policy_and_evidence_transport"
        ),
        "src/portfolio_bot/final_test/production_results.py": (
            "stage13_empirical_result_assembler_v3_terminal_sensitivity_assembly"
        ),
        "src/portfolio_bot/final_test/production_runtime.py": (
            "retired_identity_inherited_risk_spawn_receipt_and_terminal_observation_runtime"
        ),
        "src/portfolio_bot/final_test/publication.py": (
            "terminal_sensitivity_independent_reconciliation_and_rule_restoring_publication_contract"
        ),
        "src/portfolio_bot/final_test/results.py": (
            "terminal_sensitivity_result_schema_and_report_input_contract"
        ),
        "src/portfolio_bot/final_test/terminal_event_overlay.py": (
            "substantive_economic_terminal_event_treatment_and_sensitivity_source_authentication"
        ),
        "src/portfolio_bot/final_test/terminal_event_sensitivity.py": (
            "reporting_only_terminal_event_counterfactual_engine"
        ),
        "src/portfolio_bot/selectors/preprocessing.py": (
            "rule_restoring_reduced_registry_global_ordinal_preservation"
        ),
        "src/portfolio_bot/validation/corporate_actions.py": (
            "terminal_sensitivity_policy_and_evidence_action_transport"
        ),
        "src/portfolio_bot/validation/empirical_selector_backend.py": (
            "non_economic_spawn_receipt_transport_correction"
        ),
        "src/portfolio_bot/validation/session_authority.py": (
            "authoritative_observed_return_session_overlay"
        ),
    }
    expected_implementation_paths = set(expected_implementation_classifications)
    focused_test_paths = {
        str(item.get("path"))
        for item in focused_test_members
        if isinstance(item, Mapping)
    }
    expected_focused_test_paths = {
        "tests/unit/conftest.py",
        "tests/unit/test_final_input_construction.py",
        "tests/unit/test_final_input_materialization.py",
        "tests/unit/test_final_raw_source_publication.py",
        "tests/unit/test_final_session_authority.py",
        "tests/unit/test_final_source_builder.py",
        "tests/unit/test_final_terminal_event_overlay.py",
        "tests/unit/test_final_terminal_event_sensitivity.py",
        "tests/unit/test_final_test_economic_kernel.py",
        "tests/unit/test_final_test_production_results.py",
        "tests/unit/test_final_test_production_runtime.py",
        "tests/unit/test_final_test_publication.py",
        "tests/unit/test_final_test_results.py",
        "tests/unit/test_numerical_materiality.py",
        "tests/unit/test_stage11_empirical_execution.py",
        "tests/unit/test_stage12_numerical_materiality_amendment.py",
        "tests/unit/test_stage9_selector_preprocessing.py",
    }
    if (
        len(implementation_paths) != len(implementation_members)
        or implementation_paths != expected_implementation_paths
        or len(focused_test_paths) != len(focused_test_members)
        or focused_test_paths != expected_focused_test_paths
    ):
        raise Stage12FreezeError("Stage 12 v12 consolidated member paths differ")
    for member, label in (
        *(
            (item, "consolidated implementation member")
            for item in implementation_members
        ),
        *((item, "consolidated focused-test member") for item in focused_test_members),
    ):
        if not isinstance(member, Mapping):
            raise Stage12FreezeError(f"{label} is not an object")
        try:
            member_path = confined_relative_path(
                project_root,
                member.get("path"),
                file_required=True,
                label=label,
            )
            member_record = stable_file(member_path, trusted_root=project_root)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(f"{label} path is unsafe") from exc
        if member_record.sha256 != member.get("sha256"):
            raise Stage12FreezeError(f"{label} SHA-256 differs")
    for member in implementation_members:
        path = str(member.get("path"))
        if (
            member.get("classification")
            != expected_implementation_classifications[path]
        ):
            raise Stage12FreezeError(
                f"Stage 12 v12 consolidated member classification differs: {path}"
            )
    approval_boundary = _as_mapping(
        consolidation.get("access_and_approval_boundary"),
        "Stage 12 v12 access and approval boundary is missing",
    )
    if approval_boundary != {
        "stage11_recomputed": False,
        "historical_stage11_or_stage12_receipts_rewritten": False,
        "stage13_execution_authorized_by_this_freeze": False,
        "final_period_access_authorized_by_this_freeze": False,
        "fresh_binding_specific_stage13_approval_required": True,
        "standing_user_instruction_may_be_bound_after_v12_verification": True,
        "additional_user_prompt_required_after_v12_verification": False,
        "fresh_stage13_model_fits_required": 6,
        "prior_economic_checkpoints_reusable": False,
    }:
        raise Stage12FreezeError("Stage 12 v12 access and approval boundary differs")
    inventory_authorization = _as_mapping(
        consolidation.get("inventory_delta_authorization"),
        "Stage 12 v12 inventory-delta authorization is missing",
    )
    if (
        inventory_authorization.get("prior_inventory_path")
        != _PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH
        or inventory_authorization.get("prior_inventory_sha256")
        != _PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256
        or not isinstance(inventory_authorization.get("added_paths"), list)
        or not isinstance(inventory_authorization.get("changed_paths"), list)
        or inventory_authorization.get("removed_paths") != []
        or len(set(inventory_authorization["added_paths"]))
        != len(inventory_authorization["added_paths"])
        or len(set(inventory_authorization["changed_paths"]))
        != len(inventory_authorization["changed_paths"])
    ):
        raise Stage12FreezeError("Stage 12 v12 inventory authorization differs")
    return consolidation, {
        "status": consolidation["status"],
        "classification": consolidation["classification"],
        "terminal_treatment_policy": {
            "relative_path": str(terminal["policy_path"]),
            "sha256": terminal_record.sha256,
            "size_bytes": terminal_record.size_bytes,
            "substantive_economic": True,
        },
        "standing_user_instruction_sha256": standing_instruction_sha256,
        "final_input_registry_contract": dict(
            _as_mapping(
                consolidation["final_input_registry_contract"],
                "Stage 13 final-input registry contract is missing",
            )
        ),
        "report_template_contract": dict(
            _as_mapping(
                consolidation["report_template_contract"],
                "Stage 13 report-template contract is missing",
            )
        ),
        "result_assembly_contract": dict(
            _as_mapping(
                consolidation["result_assembly_contract"],
                "Stage 13 result-assembly contract is missing",
            )
        ),
        "implementation_member_count": len(implementation_members),
        "focused_test_member_count": len(focused_test_members),
        "fresh_binding_specific_stage13_approval_required": True,
        "stage13_execution_authorized_now": False,
    }


def _validate_stage12_approval(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    binding = _as_mapping(
        policy.get("stage12_approval"), "Stage 12 approval binding is missing"
    )
    expected_binding = {
        "path": "configs/phase_06_final_test/stage_12_freeze_approval_v1.json",
        "sha256": ("59adb4639d538b13508c0253e69bb8dbdc4b70f7f6e5bdc0ab8df7df286e5678"),
    }
    _require_exact_mapping(
        binding,
        expected_binding,
        label="Stage 12 approval binding",
    )
    try:
        selected = confined_relative_path(
            project_root,
            binding["path"],
            file_required=True,
            label="Stage 12 approval receipt",
        )
        approval, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("Stage 12 approval path is unsafe") from exc
    payload = dict(approval)
    observed_payload_sha = payload.pop("payload_sha256", None)
    computed_payload_sha = sha256_bytes(canonical_json_bytes(payload))
    readiness = _as_mapping(
        approval.get("implementation_readiness_acknowledged"),
        "Stage 12 approval readiness acknowledgment is missing",
    )
    bindings = _as_mapping(
        approval.get("bindings"), "Stage 12 approval bindings are missing"
    )
    if (
        record.sha256 != binding["sha256"]
        or approval.get("schema_version") != 1
        or approval.get("approval_id") != "stage12-metadata-only-current-codex-task-v1"
        or approval.get("source") != "current_codex_task_user_message"
        or approval.get("source_message_text_sha256")
        != "883042a902a6d75e7eb6afb7819cadf93a8a30b6bf7d3ec24822ca809e48518a"
        or approval.get("scope") != "stage12_metadata_only_pre_final_freeze"
        or approval.get("approved") is not True
        or approval.get("final_period_payload_access_approved") is not False
        or approval.get("stage13_execution_approved") is not False
        or approval.get("stage11_recomputation_approved") is not False
        or approval.get("memoized_resume_required") is not True
        or approval.get(
            "negligible_numeric_buffer_may_apply_only_under_frozen_materiality_policy"
        )
        is not True
        or approval.get(
            "calibration_scope_economic_amendment_explicitly_user_acknowledged"
        )
        is not False
        or observed_payload_sha != computed_payload_sha
        or observed_payload_sha
        != "d36a587691f6fa3fd81803c4cf826874989b516a9051d4f4dad9ea64b0e76211"
        or readiness
        != {
            "specification_frozen": True,
            "final_economic_kernel_certified": True,
            "schema_bound_registration_materializer_certified": True,
            "complete_final_input_arrow_schemas_frozen": True,
            "pre_unblind_calibration_scope_amendment_frozen": True,
            "stage13_technical_gate_passed": True,
        }
        or bindings.get("methodology_source_sha256")
        != "77ac9a4a79b0f46188b56f9a9a3d4affccdbfb13feafdc0f5facc79f2bbc6e81"
        or bindings.get("methodology_freeze_receipt_sha256")
        != "cbd897684639a2ccfe1ecf266cf458ef836017b91310fed2e266a75c38808d98"
        or bindings.get("stage11_primary_build_id")
        != "stage11-validation-8d5e7f863ee093089520457e"
        or bindings.get("stage11_primary_manifest_sha256")
        != "28481a09d5f933ed97f54b7441b45e9ac728ac4616e40fb76ce2b8f5abf60801"
        or bindings.get("stage11_verification_id") != "verify-86a1e2d42b4c84f0"
        or bindings.get("stage12_policy_id") != "stage12-pre-final-freeze-policy-v1"
        or bindings.get("stage12_final_system_path")
        != policy["structured_final_system"]["path"]
        or bindings.get("stage12_final_system_sha256") != _EXPECTED_FINAL_SYSTEM_SHA256
        or bindings.get("stage12_calibration_scope_amendment_path")
        != (
            "configs/phase_06_final_test/"
            "stage_12_calibration_scope_amendment_v1.json"
        )
        or bindings.get("stage12_calibration_scope_amendment_sha256")
        != _EXPECTED_CALIBRATION_SCOPE_AMENDMENT_SHA256
        or bindings.get("stage13_input_registry_path")
        != policy["final_input_registry"]["path"]
        or bindings.get("stage13_input_registry_sha256")
        != "bfef5266ec13af8377d3950fb70534b1ade27b599797cd72bbf315578b10994c"
    ):
        raise Stage12FreezeError("Stage 12 approval contract differs")
    return approval, {
        "relative_path": str(binding["path"]),
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "payload_sha256": computed_payload_sha,
    }


def _validate_final_system(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    binding = _as_mapping(
        policy.get("structured_final_system"), "final system binding is missing"
    )
    relative = str(binding.get("path"))
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="structured final system",
        )
        value, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("structured final system path is unsafe") from exc
    if (
        record.sha256 != _EXPECTED_FINAL_SYSTEM_SHA256
        or value.get("schema_version") != 1
        or value.get("system_id") != "portfolio-bot-final-system-v1.1-prefinal"
        or value.get("methodology_version") != "v1"
        or value.get("pre_unblind_calibration_scope_amendment")
        != "stage12-prefinal-shared-calibration-scope-v1"
    ):
        raise Stage12FreezeError("final system identity differs")
    models = _as_mapping(value.get("selected_models"), "selected models are missing")
    expected_models = {
        "neural_network_config_id": "nn_001",
        "xgboost_config_id": "xgb_core_038_g10",
        "conditional_elastic_net_config_id": "enet_r004_l005",
        "final_seed_order": [11, 23, 47, 71, 101],
        "neural_network_weight": 0.5,
        "xgboost_weight": 0.5,
    }
    for name, expected in expected_models.items():
        if models.get(name) != expected:
            raise Stage12FreezeError(f"selected model binding differs: {name}")
    learning = _as_mapping(
        value.get("features_and_learning"),
        "features and learning policy is missing",
    )
    expected_calibration_policy = {
        "calibration": "zero_intercept_huber_delta_1.345_nonnegative",
        "calibration_scope_policy_id": (
            "full-feature-prefinal-and-prior-live-shared-by-strategy-v1"
        ),
        "calibration_source_scope": "full_feature_primary",
        "same_strategy_calibration_slope_shared_across_feature_scopes": True,
        "reduced_core_scores_relabelled_as_prefinal_oof": False,
        "completed_labels_only": True,
    }
    for name, expected in expected_calibration_policy.items():
        if learning.get(name) != expected:
            raise Stage12FreezeError(f"final calibration policy differs: {name}")
    chronology = _as_mapping(value.get("chronology"), "chronology is missing")
    chronology_expected = {
        "primary_final_holding_months": "2020-01 through 2025-08",
        "primary_final_month_count": 68,
        "core_extension_holding_months": "2025-09 through 2025-11",
        "core_extension_month_count": 3,
        "partial_descriptive_holding_month": "2025-12",
        "partial_descriptive_in_inference": False,
        "primary_and_core_series_may_be_spliced": False,
    }
    for name, expected in chronology_expected.items():
        if chronology.get(name) != expected:
            raise Stage12FreezeError(f"final chronology differs: {name}")
    materiality = _as_mapping(
        value.get("numerical_materiality"), "numerical materiality is missing"
    )
    if (
        materiality.get("per_position_dollar_threshold_strict_upper_bound") != 0.05
        or materiality.get("aggregate_residue_dollars_per_strategy_month_maximum")
        != 1.0
        or materiality.get(
            "schemas_keys_dates_hashes_missing_inputs_psd_checks_and_solver_certificates_buffered"
        )
        is not False
    ):
        raise Stage12FreezeError("numerical materiality boundary differs")
    readiness = _as_mapping(
        value.get("implementation_readiness"),
        "implementation readiness is missing",
    )
    if readiness != {
        "specification_frozen": True,
        "segment_checkpoint_and_publication_infrastructure_certified_synthetically": True,
        "final_economic_kernel_certified": True,
        "schema_bound_registration_materializer_certified": True,
        "complete_final_input_arrow_schemas_frozen": True,
        "pre_unblind_calibration_scope_amendment_frozen": True,
        "stage13_technical_gate_passed": True,
        "final_period_access_authorized": False,
        "blockers": _STAGE13_BLOCKERS,
    }:
        raise Stage12FreezeError("implementation readiness contract differs")
    post_unblind = _as_mapping(
        value.get("post_unblind"), "post-unblind policy is missing"
    )
    if any(bool(item) for item in post_unblind.values() if isinstance(item, bool)):
        raise Stage12FreezeError("post-unblind mutation is not fully prohibited")
    return value, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
    }


def _validate_report_templates(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> list[dict[str, Any]]:
    templates = policy.get("report_templates")
    expected_templates = [
        "configs/phase_06_final_test/stage_13_report_template_v1.md",
        "configs/phase_06_final_test/stage_13_report_template_v2.md",
        "configs/phase_06_final_test/stage_14_report_template_v1.md",
    ]
    if templates != expected_templates:
        raise Stage12FreezeError("report template registry differs")
    consolidation = _as_mapping(
        policy.get("consolidated_stage13_completion_governance"),
        "Stage 12 v12 completion consolidation is missing",
    )
    template_contract = _as_mapping(
        consolidation.get("report_template_contract"),
        "Stage 13 report-template contract is missing",
    )
    if template_contract != {
        "historical_stage13_paths": [expected_templates[0]],
        "active_stage13_path": expected_templates[1],
        "active_stage14_path": expected_templates[2],
        "active_stage13_template_version": 2,
        "historical_templates_may_be_rewritten": False,
    }:
        raise Stage12FreezeError("report template role registry differs")
    roles = {
        expected_templates[0]: ("historical_superseded_stage13", False),
        expected_templates[1]: ("active_stage13", True),
        expected_templates[2]: ("active_stage14", True),
    }
    records: list[dict[str, Any]] = []
    for relative in templates:
        try:
            selected = confined_relative_path(
                project_root,
                relative,
                file_required=True,
                label="Stage 12 report template",
            )
            record = stable_file(
                selected,
                keep_payload=True,
                trusted_root=project_root,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("report template path is unsafe") from exc
        assert record.payload is not None
        try:
            text = record.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Stage12FreezeError("report template is not UTF-8") from exc
        required_freeze_marker = (
            "DO NOT EDIT AFTER THE REPLACEMENT STAGE 12 V12 FREEZE"
            if relative == expected_templates[1]
            else "DO NOT EDIT AFTER STAGE 12 FREEZE"
        )
        if required_freeze_marker not in text or "{{" not in text or "}}" not in text:
            raise Stage12FreezeError("report template is not a closed template")
        if relative == expected_templates[1] and (
            "## Mandatory terminal-liquidation sensitivity" not in text
            or "all six strategy observations" not in text
            or "four separate one-at-a-time, reporting-only counterfactuals" not in text
        ):
            raise Stage12FreezeError(
                "active Stage 13 report template lacks terminal sensitivity"
            )
        role, active = roles[str(relative)]
        records.append(
            {
                "relative_path": str(relative),
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "role": role,
                "active": active,
            }
        )
    return records


def _validate_final_input_registry(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    binding = _as_mapping(
        policy.get("final_input_registry"), "final input registry binding is missing"
    )
    relative = str(binding.get("path"))
    try:
        selected = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label="final input registry",
        )
        value, record = load_json_object(selected, trusted_root=project_root)
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("final input registry path is unsafe") from exc
    templates = value.get("templates")
    rules = _as_mapping(
        value.get("global_rules"), "final input registry rules are missing"
    )
    expected_rules = {
        "caller_created_template_permitted": False,
        "caller_supplied_reader_permitted": False,
        "source_content_hash_must_be_independently_observed": True,
        "arrow_schema_must_match_registered_schema": True,
        "holding_month_values_must_equal_requested_domain": True,
        "projection_must_be_exact_requested_columns": True,
        "primary_core_partial_splicing_permitted": False,
        "partial_december_outcome_role_permitted": False,
    }
    if (
        value.get("schema_version") != 1
        or value.get("registry_id") != "stage13-final-input-registry-v1"
        or value.get("status") != "FROZEN_BEFORE_FINAL_UNBLIND"
        or not isinstance(templates, list)
        or len(templates) != len(_EXPECTED_FINAL_INPUT_REGISTRATIONS)
        or len({str(item.get("registration_name")) for item in templates})
        != len(templates)
        or dict(rules) != expected_rules
    ):
        raise Stage12FreezeError("final input registry identity differs")
    registrations = {
        str(item.get("registration_name"))
        for item in templates
        if isinstance(item, Mapping)
    }
    if registrations != _EXPECTED_FINAL_INPUT_REGISTRATIONS:
        raise Stage12FreezeError("final input registry closed name set differs")
    segments = {str(item.get("segment")) for item in templates}
    if not {
        "primary_full",
        "core_complete",
        "partial_december_descriptive",
    }.issubset(segments):
        raise Stage12FreezeError("final input registry segment coverage differs")
    for raw_template in templates:
        template = _as_mapping(raw_template, "final input registry template is invalid")
        if set(template) != {
            "registration_name",
            "role",
            "segment",
            "month_domain",
            "required_columns",
            "arrow_schema",
        }:
            raise Stage12FreezeError(
                "final input registry template lacks an exact Arrow schema"
            )
        columns = template.get("required_columns")
        schema = template.get("arrow_schema")
        if (
            not isinstance(columns, list)
            or not columns
            or len(columns) != len(set(columns))
            or not isinstance(schema, list)
            or len(schema) != len(columns)
        ):
            raise Stage12FreezeError("final input registry Arrow schema differs")
        schema_names: list[str] = []
        for raw_field in schema:
            field = _as_mapping(raw_field, "registered Arrow field is invalid")
            name = str(field.get("name"))
            if (
                set(field) != {"name", "type", "nullable"}
                or field.get("type") not in _SUPPORTED_ARROW_TYPES
                or type(field.get("nullable")) is not bool
            ):
                raise Stage12FreezeError(
                    f"final input registry Arrow field differs: {name}"
                )
            schema_names.append(name)
        if schema_names != columns:
            raise Stage12FreezeError("final input registry Arrow schema/order differs")
    consolidation = _as_mapping(
        policy.get("consolidated_stage13_completion_governance"),
        "Stage 12 v12 completion consolidation is missing",
    )
    registry_contract = _as_mapping(
        consolidation.get("final_input_registry_contract"),
        "Stage 13 final-input registry contract is missing",
    )
    total_arrow_column_count = sum(
        len(item["arrow_schema"]) for item in templates if isinstance(item, Mapping)
    )
    if (
        registry_contract.get("path") != relative
        or registry_contract.get("registry_id") != value.get("registry_id")
        or registry_contract.get("schema_version") != value.get("schema_version")
        or registry_contract.get("template_count") != len(templates)
        or len(templates) != 32
        or registry_contract.get("total_arrow_column_count") != total_arrow_column_count
        or total_arrow_column_count != 1210
        or registry_contract.get("classification")
        != "authorized_terminal_sensitivity_evidence_transport_schema"
    ):
        raise Stage12FreezeError("final input registry v12 contract differs")
    return value, {
        "relative_path": relative,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "template_count": len(templates),
        "total_arrow_column_count": total_arrow_column_count,
    }


def build_replacement_inventory_delta(
    *,
    project_root: Path,
    project_inventory: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove the replacement freeze contains only its exact support delta."""

    selected_policy = policy
    if selected_policy is None:
        selected_policy, _ = _load_policy(DEFAULT_POLICY, project_root=project_root)
    consolidation = _as_mapping(
        selected_policy.get("consolidated_stage13_completion_governance"),
        "Stage 12 v12 completion consolidation is missing",
    )
    delta_authorization = _as_mapping(
        consolidation.get("inventory_delta_authorization"),
        "Stage 12 v12 inventory-delta authorization is missing",
    )

    try:
        prior_path = confined_relative_path(
            project_root,
            _PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH,
            file_required=True,
            label="prior immutable Stage 12 file inventory",
        )
        prior, prior_record = load_json_object(
            prior_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "prior immutable Stage 12 file inventory path is unsafe"
        ) from exc
    prior_files = prior.get("files")
    current_files = project_inventory.get("files")
    if (
        prior_record.sha256 != _PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256
        or prior.get("schema_version") != 1
        or prior.get("file_count") != 610
        or prior.get("tree_sha256")
        != "77a08b63b34e93a9750e0e820b6329b3817cbf50f5970999d0087a593cd6c032"
        or not isinstance(prior_files, list)
        or not isinstance(current_files, list)
    ):
        raise Stage12FreezeError("prior Stage 12 file inventory identity differs")
    assert isinstance(prior_files, list)
    assert isinstance(current_files, list)
    if prior.get("tree_sha256") != sha256_bytes(canonical_json_bytes(prior_files)):
        raise Stage12FreezeError("prior Stage 12 file inventory digest differs")
    if project_inventory.get("file_count") != len(
        current_files
    ) or project_inventory.get("tree_sha256") != sha256_bytes(
        canonical_json_bytes(current_files)
    ):
        raise Stage12FreezeError("replacement Stage 12 file inventory digest differs")
    prior_by_path = {
        str(item["relative_path"]): item
        for item in prior_files
        if isinstance(item, Mapping)
    }
    current_by_path = {
        str(item["relative_path"]): item
        for item in current_files
        if isinstance(item, Mapping)
    }
    if len(prior_by_path) != len(prior_files) or len(current_by_path) != len(
        current_files
    ):
        raise Stage12FreezeError("replacement inventory contains duplicate paths")
    prior_paths = set(prior_by_path)
    current_paths = set(current_by_path)
    added_paths = current_paths - prior_paths
    removed_paths = prior_paths - current_paths
    changed_paths = {
        path
        for path in prior_paths & current_paths
        if prior_by_path[path].get("sha256") != current_by_path[path].get("sha256")
    }
    expected_added_paths = set(delta_authorization.get("added_paths", []))
    expected_changed_paths = set(delta_authorization.get("changed_paths", []))
    expected_removed_paths = set(delta_authorization.get("removed_paths", []))
    if (
        delta_authorization.get("prior_inventory_path")
        != _PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH
        or delta_authorization.get("prior_inventory_sha256")
        != _PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256
        or len(expected_added_paths) != len(delta_authorization.get("added_paths", []))
        or len(expected_changed_paths)
        != len(delta_authorization.get("changed_paths", []))
        or len(expected_removed_paths)
        != len(delta_authorization.get("removed_paths", []))
        or added_paths != expected_added_paths
        or changed_paths != expected_changed_paths
        or removed_paths != expected_removed_paths
    ):
        raise Stage12FreezeError(
            "prior-to-replacement Stage 12 inventory delta differs"
        )
    prohibited_exact = {
        "configs/phase_06_final_test/stage_12_final_system_v1.json",
    }
    prohibited_prefixes = ("src/portfolio_bot/models/",)
    authorized_allocation_path = "src/portfolio_bot/allocation/solver.py"
    frozen_stage11_adapter_path = "scripts/stage_11_allocation_numerical_adapter.py"
    frozen_stage11_adapter_sha256 = (
        "fb5a2afd09a2e331b718d8d7f47842952e3d4aef41bcff7affab797fb49fa838"
    )
    all_delta_paths = added_paths | changed_paths | removed_paths
    if (
        "configs/phase_06_final_test/stage_13_input_registry_v1.json"
        not in changed_paths
        or authorized_allocation_path not in changed_paths
        or "src/portfolio_bot/final_test/production_results.py" not in changed_paths
        or "configs/phase_06_final_test/stage_13_report_template_v2.md"
        not in added_paths
        or "src/portfolio_bot/final_test/terminal_event_sensitivity.py"
        not in added_paths
        or "tests/unit/test_final_terminal_event_sensitivity.py" not in added_paths
    ):
        raise Stage12FreezeError(
            "replacement inventory omits the v12 sensitivity/reporting surface"
        )
    if any(
        path in prohibited_exact
        or path.startswith(prohibited_prefixes)
        or (
            path.startswith("src/portfolio_bot/allocation/")
            and path != authorized_allocation_path
        )
        or (
            path.startswith("src/portfolio_bot/selectors/")
            and path != "src/portfolio_bot/selectors/preprocessing.py"
        )
        for path in all_delta_paths
    ):
        raise Stage12FreezeError(
            "replacement inventory changes unauthorized allocation/model code"
        )
    if (
        frozen_stage11_adapter_path not in prior_by_path
        or frozen_stage11_adapter_path not in current_by_path
        or prior_by_path[frozen_stage11_adapter_path].get("sha256")
        != frozen_stage11_adapter_sha256
        or current_by_path[frozen_stage11_adapter_path].get("sha256")
        != frozen_stage11_adapter_sha256
        or frozen_stage11_adapter_path in all_delta_paths
    ):
        raise Stage12FreezeError(
            "replacement inventory changes the frozen Stage 11 allocation adapter"
        )
    for path in prohibited_exact:
        if (
            path not in prior_by_path
            or path not in current_by_path
            or prior_by_path[path].get("sha256") != current_by_path[path].get("sha256")
        ):
            raise Stage12FreezeError(
                f"replacement inventory changes frozen authority: {path}"
            )
    changed = [
        {
            "relative_path": path,
            "prior_sha256": str(prior_by_path[path]["sha256"]),
            "replacement_sha256": str(current_by_path[path]["sha256"]),
            "domain": str(current_by_path[path]["domain"]),
            "required_for_stage13": bool(current_by_path[path]["required_for_stage13"]),
            "required_for_stage14": bool(current_by_path[path]["required_for_stage14"]),
        }
        for path in sorted(changed_paths)
    ]
    added = [
        {
            "relative_path": path,
            "replacement_sha256": str(current_by_path[path]["sha256"]),
            "domain": str(current_by_path[path]["domain"]),
            "required_for_stage13": bool(current_by_path[path]["required_for_stage13"]),
            "required_for_stage14": bool(current_by_path[path]["required_for_stage14"]),
        }
        for path in sorted(added_paths)
    ]
    return {
        "schema_version": 1,
        "status": "EXACT_POST_UNBLIND_REPLACEMENT_INVENTORY_DELTA_AUTHENTICATED",
        "prior_inventory": {
            "relative_path": _PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH,
            "sha256": prior_record.sha256,
            "size_bytes": prior_record.size_bytes,
            "file_count": int(prior["file_count"]),
            "tree_sha256": str(prior["tree_sha256"]),
        },
        "replacement_inventory": {
            "file_count": int(project_inventory["file_count"]),
            "tree_sha256": str(project_inventory["tree_sha256"]),
        },
        "changed_member_count": len(changed),
        "changed_members": changed,
        "added_member_count": len(added),
        "added_members": added,
        "removed_member_count": len(removed_paths),
        "removed_members": sorted(removed_paths),
        "semantic_checks": {
            "policy_bound_changed_path_allowlist_matched": True,
            "policy_bound_added_path_allowlist_matched": True,
            "policy_bound_removed_path_allowlist_matched": True,
            "frozen_final_system_unchanged": True,
            "final_input_registry_1210_column_sensitivity_transport_authorized": True,
            "historical_stage13_v1_report_template_preserved": True,
            "active_stage13_v2_report_template_added": True,
            "stage13_empirical_result_assembler_v3_bound": True,
            "substantive_terminal_treatment_explicitly_authorized_by_v12_policy": True,
            "unpublished_v11_retained_as_non_authoritative_history": True,
            "model_code_unchanged": True,
            "allocation_change_limited_to_rule_preserving_solver_api_compatibility": True,
            "stage11_allocation_adapter_preserved_byte_exact": True,
            "selector_change_limited_to_authorized_global_ordinal_preservation": True,
            "new_member_hashes_bound_by_replacement_project_inventory": True,
        },
    }


def build_source_drift_record(
    *,
    project_root: Path,
    project_inventory: Mapping[str, Any],
    amendment_path: Path = MATERIALITY_AMENDMENT,
    post_unblind_amendment_path: Path = POST_UNBLIND_TECHNICAL_AMENDMENT,
    zero_row_amendment_path: Path = (POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT),
    december_amendment_path: Path = (
        POST_UNBLIND_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT
    ),
    entry_market_amendment_path: Path = (
        POST_UNBLIND_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT
    ),
    accounting_transport_amendment_path: Path = (
        POST_UNBLIND_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT
    ),
    logical_fingerprint_transport_amendment_path: Path = (
        POST_UNBLIND_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT
    ),
    factor_covariance_ordinal_basis_alignment_amendment_path: Path = (
        POST_UNBLIND_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT
    ),
    equal_weight_missing_cost_gate_amendment_path: Path = (
        POST_UNBLIND_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT
    ),
) -> dict[str, Any]:
    """Authenticate every frozen Stage 11-to-12 source-transition layer."""

    try:
        selected_amendment = confined_existing_path(
            amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 materiality amendment",
        )
        amendment, amendment_file = load_json_object(
            selected_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("materiality amendment path is unsafe") from exc
    try:
        selected_post_unblind_amendment = confined_existing_path(
            post_unblind_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 post-unblind technical amendment",
        )
        post_unblind_amendment, post_unblind_amendment_file = load_json_object(
            selected_post_unblind_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "post-unblind technical amendment path is unsafe"
        ) from exc
    post_unblind_transition = _as_mapping(
        post_unblind_amendment.get("source_transition"),
        "post-unblind source transition is missing",
    )
    if (
        post_unblind_amendment_file.sha256
        != _EXPECTED_POST_UNBLIND_TECHNICAL_AMENDMENT_SHA256
        or post_unblind_amendment.get("amendment_id")
        != "stage12-post-unblind-auxiliary-nullability-transport-v1"
        or post_unblind_amendment.get("status")
        != "STAGE12_POST_UNBLIND_TECHNICAL_AMENDMENT_AUTHORIZED"
        or post_unblind_transition.get("exact_changed_stage11_bound_member_count") != 1
    ):
        raise Stage12FreezeError("post-unblind source transition differs")
    try:
        selected_zero_row_amendment = confined_existing_path(
            zero_row_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 zero-row merge provenance amendment",
        )
        zero_row_amendment, zero_row_amendment_file = load_json_object(
            selected_zero_row_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "zero-row merge provenance amendment path is unsafe"
        ) from exc
    zero_row_transition = _as_mapping(
        zero_row_amendment.get("source_transition"),
        "zero-row merge provenance source transition is missing",
    )
    zero_row_members = zero_row_transition.get("members")
    if (
        zero_row_amendment_file.sha256
        != _EXPECTED_POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_SHA256
        or zero_row_amendment.get("amendment_id")
        != "stage12-post-unblind-zero-row-merge-provenance-v1"
        or zero_row_amendment.get("status")
        != "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED"
        or zero_row_transition.get("exact_changed_runtime_member_count") != 2
        or not isinstance(zero_row_members, list)
        or len(zero_row_members) != 2
    ):
        raise Stage12FreezeError("zero-row merge provenance source transition differs")
    try:
        selected_december_amendment = confined_existing_path(
            december_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 December market-input scope-partition amendment",
        )
        december_amendment, december_amendment_file = load_json_object(
            selected_december_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "December market-input scope-partition amendment path is unsafe"
        ) from exc
    december_transition = _as_mapping(
        december_amendment.get("source_transition"),
        "December market-input source transition is missing",
    )
    december_members = december_transition.get("members")
    if (
        december_amendment_file.sha256
        != _EXPECTED_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_SHA256
        or december_amendment.get("amendment_id")
        != "stage12-post-unblind-december-market-input-scope-partition-v1"
        or december_amendment.get("status")
        != "STAGE12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_AUTHORIZED"
        or december_transition.get("exact_changed_existing_runtime_member_count") != 2
        or december_transition.get("exact_added_runtime_member_count") != 1
        or not isinstance(december_members, list)
        or len(december_members) != 3
    ):
        raise Stage12FreezeError("December market-input source transition differs")
    try:
        selected_entry_market_amendment = confined_existing_path(
            entry_market_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 entry-market alignment amendment",
        )
        entry_market_amendment, entry_market_amendment_file = load_json_object(
            selected_entry_market_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "entry-market alignment amendment path is unsafe"
        ) from exc
    entry_market_transition = _as_mapping(
        entry_market_amendment.get("source_transition"),
        "entry-market alignment source transition is missing",
    )
    entry_market_members = entry_market_transition.get("members")
    if (
        entry_market_amendment_file.sha256
        != _EXPECTED_ENTRY_MARKET_SCOPE_ALIGNMENT_AMENDMENT_SHA256
        or entry_market_amendment.get("amendment_id")
        != "stage12-post-unblind-entry-market-canonicalization-scope-alignment-v1"
        or entry_market_amendment.get("status")
        != "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT_AUTHORIZED"
        or entry_market_transition.get("exact_changed_runtime_member_count") != 2
        or not isinstance(entry_market_members, list)
        or len(entry_market_members) != 2
    ):
        raise Stage12FreezeError("entry-market alignment source transition differs")
    try:
        selected_accounting_amendment = confined_existing_path(
            accounting_transport_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 accounting missing-input transport amendment",
        )
        accounting_amendment, accounting_amendment_file = load_json_object(
            selected_accounting_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "accounting missing-input transport amendment path is unsafe"
        ) from exc
    accounting_transition = _as_mapping(
        accounting_amendment.get("source_transition"),
        "accounting transport source transition is missing",
    )
    accounting_members = accounting_transition.get("members")
    if (
        accounting_amendment_file.sha256
        != _EXPECTED_ACCOUNTING_TRANSPORT_AMENDMENT_SHA256
        or accounting_amendment.get("amendment_id")
        != "stage12-post-unblind-accounting-missing-execution-cost-transport-v1"
        or accounting_amendment.get("status")
        != (
            "STAGE12_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT_"
            "AUTHORIZED"
        )
        or accounting_transition.get("exact_changed_runtime_member_count") != 2
        or not isinstance(accounting_members, list)
        or accounting_members
        != [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": _EXPECTED_ACCOUNTING_REPLAY_PRE_SHA256,
                "post_amendment_sha256": _EXPECTED_ACCOUNTING_REPLAY_POST_SHA256,
            },
            {
                "path": "src/portfolio_bot/final_test/input_materialization.py",
                "pre_amendment_sha256": _EXPECTED_ACCOUNTING_MATERIALIZER_PRE_SHA256,
                "post_amendment_sha256": _EXPECTED_ACCOUNTING_MATERIALIZER_POST_SHA256,
            },
        ]
    ):
        raise Stage12FreezeError("accounting transport source transition differs")
    try:
        selected_logical_fingerprint_amendment = confined_existing_path(
            logical_fingerprint_transport_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 logical-fingerprint transport amendment",
        )
        logical_fingerprint_amendment, logical_fingerprint_amendment_file = (
            load_json_object(
                selected_logical_fingerprint_amendment,
                trusted_root=project_root,
            )
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "logical-fingerprint transport amendment path is unsafe"
        ) from exc
    logical_fingerprint_transition = _as_mapping(
        logical_fingerprint_amendment.get("source_transition"),
        "logical-fingerprint source transition is missing",
    )
    logical_fingerprint_members = logical_fingerprint_transition.get("members")
    if (
        logical_fingerprint_amendment_file.sha256
        != _EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SHA256
        or logical_fingerprint_amendment.get("amendment_id")
        != "stage12-post-unblind-logical-fingerprint-transport-v1"
        or logical_fingerprint_amendment.get("status")
        != "STAGE12_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_AUTHORIZED"
        or logical_fingerprint_transition.get("exact_changed_runtime_member_count") != 1
        or logical_fingerprint_members
        != [
            {
                "path": "src/portfolio_bot/final_test/final_source_replay.py",
                "pre_amendment_sha256": (
                    _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_PRE_SHA256
                ),
                "post_amendment_sha256": (
                    _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
                ),
            }
        ]
    ):
        raise Stage12FreezeError(
            "logical-fingerprint transport source transition differs"
        )
    try:
        selected_factor_covariance_amendment = confined_existing_path(
            factor_covariance_ordinal_basis_alignment_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 factor-covariance ordinal-basis amendment",
        )
        factor_covariance_amendment, factor_covariance_amendment_file = (
            load_json_object(
                selected_factor_covariance_amendment,
                trusted_root=project_root,
            )
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "factor-covariance ordinal-basis amendment path is unsafe"
        ) from exc
    factor_covariance_transition = _as_mapping(
        factor_covariance_amendment.get("source_transition"),
        "factor-covariance ordinal-basis source transition is missing",
    )
    factor_covariance_members = factor_covariance_transition.get("members")
    expected_factor_covariance_members = [
        {
            "path": "src/portfolio_bot/final_test/final_input_construction.py",
            "pre_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_COMPILER_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_COMPILER_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/production_runtime.py",
            "pre_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_RUNTIME_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_FACTOR_COVARIANCE_RUNTIME_POST_SHA256,
        },
    ]
    if (
        factor_covariance_amendment_file.sha256
        != _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_SHA256
        or sha256_bytes(canonical_json_bytes(factor_covariance_amendment))
        != _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_SEMANTIC_SHA256
        or factor_covariance_amendment.get("amendment_id")
        != "stage12-post-unblind-factor-covariance-ordinal-basis-alignment-v1"
        or factor_covariance_amendment.get("status")
        != ("STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_" "AUTHORIZED")
        or factor_covariance_transition.get("exact_changed_runtime_member_count") != 2
        or factor_covariance_members != expected_factor_covariance_members
    ):
        raise Stage12FreezeError(
            "factor-covariance ordinal-basis source transition differs"
        )
    try:
        selected_equal_weight_amendment = confined_existing_path(
            equal_weight_missing_cost_gate_amendment_path,
            trusted_root=project_root,
            file_required=True,
            label="Stage 12 equal-weight missing-cost gate amendment",
        )
        equal_weight_amendment, equal_weight_amendment_file = load_json_object(
            selected_equal_weight_amendment,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError(
            "equal-weight missing-cost gate amendment path is unsafe"
        ) from exc
    equal_weight_transition = _as_mapping(
        equal_weight_amendment.get("source_transition"),
        "equal-weight missing-cost gate source transition is missing",
    )
    equal_weight_members = equal_weight_transition.get("members")
    expected_equal_weight_members = [
        {
            "path": "src/portfolio_bot/final_test/economic_kernel.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_KERNEL_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_KERNEL_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/publication.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_PUBLICATION_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_PUBLICATION_POST_SHA256,
        },
        {
            "path": "src/portfolio_bot/final_test/production_runtime.py",
            "pre_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_RUNTIME_PRE_SHA256,
            "post_amendment_sha256": _EXPECTED_EQUAL_WEIGHT_RUNTIME_POST_SHA256,
        },
    ]
    if (
        equal_weight_amendment_file.sha256
        != _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SHA256
        or sha256_bytes(canonical_json_bytes(equal_weight_amendment))
        != _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SEMANTIC_SHA256
        or equal_weight_amendment.get("amendment_id")
        != "stage12-post-unblind-equal-weight-missing-cost-gate-v1"
        or equal_weight_amendment.get("status")
        != "STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED"
        or equal_weight_transition.get("exact_changed_runtime_member_count") != 3
        or equal_weight_members != expected_equal_weight_members
    ):
        raise Stage12FreezeError(
            "equal-weight missing-cost gate source transition differs"
        )
    transition = _as_mapping(
        amendment.get("source_transition"), "source transition is missing"
    )
    members = transition.get("members")
    if not isinstance(members, list) or len(members) != int(
        transition.get("changed_stage11_bound_member_count", -1)
    ) + int(transition.get("new_stage12_member_count", -1)):
        raise Stage12FreezeError("source transition member list differs")
    stage11_pointer_path = confined_relative_path(
        project_root,
        "artifacts/phase_05_validation/LATEST_STAGE_11_EXECUTION_VERIFIED.json",
        file_required=True,
        label="Stage 11 verified pointer",
    )
    stage11_pointer, _ = load_json_object(
        stage11_pointer_path,
        trusted_root=project_root,
    )
    boundary = _as_mapping(
        amendment.get("authority_boundary"), "Stage 11 authority boundary is missing"
    )
    expected_build_id = str(boundary.get("stage11_primary_build_id"))
    if (
        stage11_pointer.get("primary_build_id") != expected_build_id
        or stage11_pointer.get("verification_id")
        != boundary.get("stage11_verification_id")
        or stage11_pointer.get("difference_count")
        != boundary.get("stage11_difference_count")
    ):
        raise Stage12FreezeError("Stage 11 pointer identity differs")
    external_root = Path.home() / "Desktop/Thesis"
    expected_bundle = (
        external_root
        / "interim/phase_05_validation/stage_11_execution_v1/execution_results/primary"
        / expected_build_id
    ).absolute()
    supplied_bundle = Path(str(stage11_pointer.get("primary_bundle_path"))).absolute()
    if supplied_bundle != expected_bundle:
        raise Stage12FreezeError("Stage 11 primary bundle path differs")
    try:
        bundle_path = confined_existing_path(
            supplied_bundle,
            trusted_root=external_root,
            file_required=False,
            label="Stage 11 primary bundle",
        )
        manifest_path = confined_relative_path(
            bundle_path,
            "manifest.json",
            file_required=True,
            label="Stage 11 primary manifest",
        )
        manifest_file = stable_file(
            manifest_path,
            keep_payload=True,
            trusted_root=external_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("Stage 11 primary manifest path is unsafe") from exc
    if manifest_file.sha256 != stage11_pointer.get("primary_manifest_sha256"):
        raise Stage12FreezeError("Stage 11 primary manifest SHA-256 differs")
    if manifest_file.sha256 != boundary.get("stage11_primary_manifest_sha256"):
        raise Stage12FreezeError("Stage 11 amendment manifest binding differs")
    assert manifest_file.payload is not None
    try:
        manifest = json.loads(manifest_file.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12FreezeError("Stage 11 primary manifest is invalid") from exc
    old_hashes = _as_mapping(
        manifest.get("implementation_code_hashes"),
        "Stage 11 implementation inventory is missing",
    )
    if len(old_hashes) != int(transition.get("stage11_bound_member_count", -1)):
        raise Stage12FreezeError("Stage 11 implementation count differs")
    current_files = {
        str(item["relative_path"]): str(item["sha256"])
        for item in project_inventory.get("files", [])
        if isinstance(item, Mapping)
    }
    observed_changes: list[dict[str, Any]] = []
    historical_metadata_files_opened = 0
    for path, old_sha in old_hashes.items():
        current_sha = current_files.get(str(path))
        if current_sha is None:
            if str(path).startswith("private_clean_portfolio_candidate/"):
                relative_clean = Path(str(path)).relative_to(
                    "private_clean_portfolio_candidate"
                )
                historical_path = (
                    bundle_path.parents[1]
                    / "clean_portfolio_candidates"
                    / relative_clean
                )
            else:
                historical_path = project_root / str(path)
            if (
                not str(path).startswith(
                    ("artifacts/", "private_clean_portfolio_candidate/")
                )
                or historical_path.suffix != ".json"
            ):
                raise Stage12FreezeError(f"Stage 11 member is absent now: {path}")
            try:
                trusted = (
                    project_root
                    if str(path).startswith("artifacts/")
                    else external_root
                )
                safe_historical = confined_existing_path(
                    historical_path,
                    trusted_root=trusted,
                    file_required=True,
                    label="historical Stage 11 metadata member",
                )
                current_sha = stable_file(
                    safe_historical,
                    trusted_root=trusted,
                ).sha256
                historical_metadata_files_opened += 1
            except Stage12InventoryError as exc:
                raise Stage12FreezeError(
                    f"Stage 11 member path is unsafe: {path}"
                ) from exc
        if current_sha != old_sha:
            observed_changes.append(
                {
                    "path": str(path),
                    "stage11_sealed_sha256": str(old_sha),
                    "stage12_sha256": current_sha,
                }
            )
    expected_changes = {
        str(item["path"]): (
            item.get("stage11_sealed_sha256"),
            item.get("stage12_sha256"),
        )
        for item in members
        if item.get("stage11_sealed_sha256") is not None
    }
    overlay_path = str(post_unblind_transition.get("adapter_path"))
    overlay_before = post_unblind_transition.get("pre_amendment_sha256")
    overlay_intermediate = post_unblind_transition.get("post_amendment_sha256")
    zero_row_by_path = {
        str(item.get("path")): item
        for item in zero_row_members
        if isinstance(item, Mapping)
    }
    zero_row_adapter = zero_row_by_path.get(overlay_path)
    zero_row_replay_path = "src/portfolio_bot/final_test/final_source_replay.py"
    zero_row_replay = zero_row_by_path.get(zero_row_replay_path)
    december_by_path = {
        str(item.get("path")): item
        for item in december_members
        if isinstance(item, Mapping)
    }
    december_replay = december_by_path.get(zero_row_replay_path)
    december_market_path = "src/portfolio_bot/validation/validation_market_extension.py"
    december_market = december_by_path.get(december_market_path)
    december_rebind = december_by_path.get(
        "src/portfolio_bot/final_test/source_checkpoint_rebind.py"
    )
    entry_market_by_path = {
        str(item.get("path")): item
        for item in entry_market_members
        if isinstance(item, Mapping)
    }
    entry_market_replay = entry_market_by_path.get(zero_row_replay_path)
    entry_market_extension = entry_market_by_path.get(december_market_path)
    overlay_after = (
        zero_row_adapter.get("post_amendment_sha256")
        if isinstance(zero_row_adapter, Mapping)
        else None
    )
    if (
        overlay_path
        != "src/portfolio_bot/validation/validation_feature_input_adapter.py"
        or overlay_before != _EXPECTED_PRE_AMENDMENT_ADAPTER_SHA256
        or overlay_intermediate != _EXPECTED_POST_AMENDMENT_ADAPTER_SHA256
        or post_unblind_transition.get("permitted_changed_stage11_bound_members")
        != [overlay_path]
        or overlay_path in expected_changes
        or set(zero_row_by_path) != {overlay_path, zero_row_replay_path}
        or not isinstance(zero_row_adapter, Mapping)
        or zero_row_adapter.get("pre_amendment_sha256")
        != _EXPECTED_ZERO_ROW_ADAPTER_PRE_SHA256
        or overlay_after != _EXPECTED_ZERO_ROW_ADAPTER_POST_SHA256
        or not isinstance(zero_row_replay, Mapping)
        or zero_row_replay.get("pre_amendment_sha256")
        != _EXPECTED_ZERO_ROW_REPLAY_PRE_SHA256
        or zero_row_replay.get("post_amendment_sha256")
        != _EXPECTED_ZERO_ROW_REPLAY_POST_SHA256
        or set(december_by_path)
        != {
            zero_row_replay_path,
            december_market_path,
            "src/portfolio_bot/final_test/source_checkpoint_rebind.py",
        }
        or not isinstance(december_replay, Mapping)
        or december_replay.get("pre_amendment_sha256")
        != _EXPECTED_ZERO_ROW_REPLAY_POST_SHA256
        or december_replay.get("post_amendment_sha256")
        != "73539ebe6b569cfacc94bce78c246766551bfbe174a2f8a8aabc4b126ba96a4f"
        or not isinstance(december_market, Mapping)
        or december_market.get("pre_amendment_sha256")
        != "1bd9a4487351f1ac3337f2d3216f946ba553ee43a6006353ed1a019d7ad09446"
        or december_market.get("post_amendment_sha256")
        != "b54e382705935b90c25372c4fc24cce7013da2beda741ae378e20113977e50f6"
        or not isinstance(december_rebind, Mapping)
        or december_rebind.get("pre_amendment_sha256") is not None
        or december_rebind.get("post_amendment_sha256")
        != "7216eb7b08b4cd0886e9ffff6497ed9a5b49c9013310c63e36a8f4e2d2d688bb"
        or set(entry_market_by_path) != {zero_row_replay_path, december_market_path}
        or not isinstance(entry_market_replay, Mapping)
        or entry_market_replay.get("pre_amendment_sha256")
        != _EXPECTED_ENTRY_MARKET_REPLAY_PRE_SHA256
        or entry_market_replay.get("post_amendment_sha256")
        != _EXPECTED_ENTRY_MARKET_REPLAY_POST_SHA256
        or not isinstance(entry_market_extension, Mapping)
        or entry_market_extension.get("pre_amendment_sha256")
        != _EXPECTED_ENTRY_MARKET_EXTENSION_PRE_SHA256
        or entry_market_extension.get("post_amendment_sha256")
        != _EXPECTED_ENTRY_MARKET_EXTENSION_POST_SHA256
        or current_files.get(zero_row_replay_path)
        != _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
        or current_files.get("src/portfolio_bot/final_test/input_materialization.py")
        != _EXPECTED_ACCOUNTING_MATERIALIZER_POST_SHA256
        or current_files.get(december_market_path)
        != entry_market_extension.get("post_amendment_sha256")
        or current_files.get("src/portfolio_bot/final_test/source_checkpoint_rebind.py")
        != december_rebind.get("post_amendment_sha256")
        or current_files.get("src/portfolio_bot/final_test/final_input_construction.py")
        != _EXPECTED_FACTOR_COVARIANCE_COMPILER_POST_SHA256
        or current_files.get("src/portfolio_bot/final_test/production_runtime.py")
        != _EXPECTED_EQUAL_WEIGHT_RUNTIME_POST_SHA256
        or current_files.get("src/portfolio_bot/final_test/economic_kernel.py")
        != _EXPECTED_EQUAL_WEIGHT_KERNEL_POST_SHA256
        or current_files.get("src/portfolio_bot/final_test/publication.py")
        != _EXPECTED_EQUAL_WEIGHT_PUBLICATION_POST_SHA256
    ):
        raise Stage12FreezeError("post-unblind layered source transition differs")
    expected_changes[overlay_path] = (overlay_before, overlay_after)
    expected_changes[december_market_path] = (
        december_market.get("pre_amendment_sha256"),
        entry_market_extension.get("post_amendment_sha256"),
    )
    observed = {
        item["path"]: (
            item["stage11_sealed_sha256"],
            item["stage12_sha256"],
        )
        for item in observed_changes
    }
    if observed != expected_changes:
        raise Stage12FreezeError("observed Stage 11-to-12 source drift differs")
    for item in members:
        path = str(item["path"])
        current_sha = current_files.get(path)
        if current_sha != item.get("stage12_sha256"):
            raise Stage12FreezeError(f"Stage 12 transition hash differs: {path}")
        if item.get("stage11_sealed_sha256") is None and path in old_hashes:
            raise Stage12FreezeError("new Stage 12 member existed in Stage 11")
    if (
        current_files.get(overlay_path) != overlay_after
        or old_hashes.get(overlay_path) != overlay_before
    ):
        raise Stage12FreezeError("post-unblind adapter source identity differs")
    if len(observed_changes) != (
        int(transition.get("changed_stage11_bound_member_count", -1)) + 2
    ):
        raise Stage12FreezeError("Stage 12 changed-member count differs")
    return {
        "schema_version": 1,
        "record_id": "stage12-equal-weight-missing-cost-gate-policy-v11",
        "status": (
            "AUTHENTICATED_RULE_RESTORING_EQUAL_WEIGHT_MISSING_COST_GATE_"
            "SOURCE_TRANSITION"
        ),
        "amendment": amendment,
        "amendment_receipt": {
            "relative_path": selected_amendment.relative_to(project_root).as_posix(),
            "sha256": amendment_file.sha256,
            "size_bytes": amendment_file.size_bytes,
        },
        "post_unblind_technical_amendment": post_unblind_amendment,
        "post_unblind_technical_amendment_receipt": {
            "relative_path": selected_post_unblind_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": post_unblind_amendment_file.sha256,
            "size_bytes": post_unblind_amendment_file.size_bytes,
        },
        "post_unblind_zero_row_merge_provenance_amendment": (zero_row_amendment),
        "post_unblind_zero_row_merge_provenance_amendment_receipt": {
            "relative_path": selected_zero_row_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": zero_row_amendment_file.sha256,
            "size_bytes": zero_row_amendment_file.size_bytes,
        },
        "post_unblind_december_market_input_scope_partition_amendment": (
            december_amendment
        ),
        "post_unblind_december_market_input_scope_partition_amendment_receipt": {
            "relative_path": selected_december_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": december_amendment_file.sha256,
            "size_bytes": december_amendment_file.size_bytes,
        },
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment": (
            entry_market_amendment
        ),
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment_receipt": {
            "relative_path": selected_entry_market_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": entry_market_amendment_file.sha256,
            "size_bytes": entry_market_amendment_file.size_bytes,
        },
        "post_unblind_accounting_missing_execution_cost_transport_amendment": (
            accounting_amendment
        ),
        "post_unblind_accounting_missing_execution_cost_transport_amendment_receipt": {
            "relative_path": selected_accounting_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": accounting_amendment_file.sha256,
            "size_bytes": accounting_amendment_file.size_bytes,
        },
        "post_unblind_logical_fingerprint_transport_amendment": (
            logical_fingerprint_amendment
        ),
        "post_unblind_logical_fingerprint_transport_amendment_receipt": {
            "relative_path": selected_logical_fingerprint_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": logical_fingerprint_amendment_file.sha256,
            "size_bytes": logical_fingerprint_amendment_file.size_bytes,
        },
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment": (
            factor_covariance_amendment
        ),
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_receipt": {
            "relative_path": selected_factor_covariance_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": factor_covariance_amendment_file.sha256,
            "semantic_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_SEMANTIC_SHA256
            ),
            "size_bytes": factor_covariance_amendment_file.size_bytes,
        },
        "post_unblind_equal_weight_missing_cost_gate_amendment": (
            equal_weight_amendment
        ),
        "post_unblind_equal_weight_missing_cost_gate_amendment_receipt": {
            "relative_path": selected_equal_weight_amendment.relative_to(
                project_root
            ).as_posix(),
            "sha256": equal_weight_amendment_file.sha256,
            "semantic_sha256": (
                _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SEMANTIC_SHA256
            ),
            "size_bytes": equal_weight_amendment_file.size_bytes,
        },
        "stage11_primary_manifest": {
            "build_id": stage11_pointer.get("primary_build_id"),
            "verification_id": stage11_pointer.get("verification_id"),
            "sha256": manifest_file.sha256,
            "bound_member_count": len(old_hashes),
        },
        "observed_changed_members": sorted(
            observed_changes, key=lambda item: item["path"]
        ),
        "authorized_post_unblind_changed_members": [
            {
                "path": overlay_path,
                "pre_amendment_sha256": overlay_before,
                "post_amendment_sha256": overlay_intermediate,
            }
        ],
        "authorized_zero_row_merge_provenance_changed_members": [
            dict(item) for item in zero_row_members
        ],
        "authorized_december_market_input_scope_partition_members": [
            dict(item) for item in december_members
        ],
        "authorized_entry_market_canonicalization_scope_alignment_changed_members": [
            dict(item) for item in entry_market_members
        ],
        "authorized_checkpoint_rebind_metadata_exception_changed_members": [
            {
                "path": zero_row_replay_path,
                "pre_exception_sha256": _EXPECTED_ENTRY_MARKET_REPLAY_POST_SHA256,
                "post_exception_sha256": (
                    _EXPECTED_CHECKPOINT_REBIND_REPLAY_POST_SHA256
                ),
                "classification": (
                    "non_economic_checkpoint_rebind_metadata_authority_"
                    "compatibility_repair"
                ),
            }
        ],
        "authorized_accounting_missing_execution_cost_transport_changed_members": [
            dict(item) for item in accounting_members
        ],
        "authorized_logical_fingerprint_transport_changed_members": [
            dict(item) for item in logical_fingerprint_members
        ],
        "authorized_factor_covariance_ordinal_basis_alignment_changed_members": [
            dict(item) for item in factor_covariance_members
        ],
        "authorized_equal_weight_missing_cost_gate_changed_members": [
            dict(item) for item in equal_weight_members
        ],
        "stage11_recomputed": False,
        "stage11_historical_metadata_files_opened": (historical_metadata_files_opened),
        "validation_performance_used": False,
        "final_period_rows_read": 0,
    }


def build_consolidated_source_drift_record(
    *,
    policy: Mapping[str, Any],
    project_root: Path,
    project_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind current v12 implementation identities to active v10 and the policy."""

    consolidation = _as_mapping(
        policy.get("consolidated_stage13_completion_governance"),
        "Stage 12 v12 completion consolidation is missing",
    )
    terminal = _as_mapping(
        consolidation.get("terminal_event_treatment"),
        "substantive terminal-event treatment is missing",
    )
    implementation_members = consolidation.get("implementation_members")
    if not isinstance(implementation_members, list) or not implementation_members:
        raise Stage12FreezeError("v12 implementation member inventory is missing")
    try:
        prior_inventory_path = confined_relative_path(
            project_root,
            _PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH,
            file_required=True,
            label="active v10 Stage 12 file inventory",
        )
        prior_inventory, prior_inventory_record = load_json_object(
            prior_inventory_path,
            trusted_root=project_root,
        )
        terminal_policy_path = confined_relative_path(
            project_root,
            terminal.get("policy_path"),
            file_required=True,
            label="substantive terminal-event treatment policy",
        )
        _, terminal_policy_record = load_json_object(
            terminal_policy_path,
            trusted_root=project_root,
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("v12 source-drift authority path is unsafe") from exc
    prior_files = prior_inventory.get("files")
    current_files = project_inventory.get("files")
    if (
        prior_inventory_record.sha256 != _PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256
        or prior_inventory.get("file_count") != 610
        or prior_inventory.get("tree_sha256")
        != "77a08b63b34e93a9750e0e820b6329b3817cbf50f5970999d0087a593cd6c032"
        or terminal_policy_record.sha256 != terminal.get("policy_sha256")
        or not isinstance(prior_files, list)
        or not isinstance(current_files, list)
    ):
        raise Stage12FreezeError("v12 source-drift authority identity differs")
    prior_by_path = {
        str(item.get("relative_path")): item
        for item in prior_files
        if isinstance(item, Mapping)
    }
    current_by_path = {
        str(item.get("relative_path")): item
        for item in current_files
        if isinstance(item, Mapping)
    }
    if len(prior_by_path) != len(prior_files) or len(current_by_path) != len(
        current_files
    ):
        raise Stage12FreezeError("v12 source-drift inventory contains duplicate paths")
    changes: list[dict[str, Any]] = []
    for member in implementation_members:
        if not isinstance(member, Mapping):
            raise Stage12FreezeError("v12 implementation member is not an object")
        path = str(member.get("path"))
        current = current_by_path.get(path)
        prior = prior_by_path.get(path)
        if current is None or current.get("sha256") != member.get("sha256"):
            raise Stage12FreezeError(f"v12 implementation identity differs: {path}")
        prior_sha = prior.get("sha256") if isinstance(prior, Mapping) else None
        if prior_sha == current.get("sha256"):
            raise Stage12FreezeError(
                f"v12 implementation member does not differ from active v10: {path}"
            )
        changes.append(
            {
                "path": path,
                "active_v10_sha256": prior_sha,
                "stage12_v12_sha256": str(current["sha256"]),
                "classification": str(member.get("classification")),
            }
        )
    changes.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "record_id": "stage12-consolidated-stage13-completion-policy-v12",
        "status": "AUTHENTICATED_CONSOLIDATED_STAGE13_COMPLETION_SOURCE_TRANSITION",
        "classification": consolidation["classification"],
        "amendment_receipt": {
            "relative_path": str(terminal["policy_path"]),
            "sha256": terminal_policy_record.sha256,
            "size_bytes": terminal_policy_record.size_bytes,
            "substantive_economic": True,
        },
        "active_v10_inventory": {
            "relative_path": _PRIOR_V10_STAGE12_FILE_INVENTORY_RELATIVE_PATH,
            "sha256": prior_inventory_record.sha256,
            "file_count": int(prior_inventory["file_count"]),
            "tree_sha256": str(prior_inventory["tree_sha256"]),
        },
        "unpublished_v11": dict(consolidation["unpublished_v11"]),
        "substantive_terminal_event_treatment": dict(terminal),
        "authorized_consolidated_changed_members": changes,
        "authorized_consolidated_changed_member_count": len(changes),
        "stage11_recomputed": False,
        "stage11_historical_metadata_files_opened": 0,
        "validation_performance_used": False,
        "final_period_rows_read": 0,
        "stage13_execution_authorized_now": False,
        "fresh_binding_specific_stage13_approval_required": True,
    }


def _validate_test_receipts(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    project_root: Path,
    project_tree_sha256: str,
    environment_sha256: str,
) -> dict[str, Any]:
    nodes = value.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise Stage12FreezeError("both Stage 12 test receipts are required")
    expected_names = {
        "focused_stage12_protocol",
        "independent_synthetic_regression",
    }
    observed_names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    observed_by_name: dict[str, dict[str, Any]] = {}
    for node in nodes:
        item = _as_mapping(node, "test receipt node is invalid")
        checkpoint_id = str(item.get("checkpoint_id"))
        relative = str(item.get("relative_path"))
        expected_relative = (
            Path(str(policy["outputs"]["artifact_root"]))
            / str(policy["outputs"]["checkpoint_namespace"])
            / checkpoint_id
        ).as_posix()
        if relative != expected_relative:
            raise Stage12FreezeError("test checkpoint namespace differs")
        try:
            path = confined_relative_path(
                project_root,
                relative,
                file_required=False,
                label="Stage 12 test checkpoint",
            )
            receipt_path = confined_relative_path(
                path,
                "receipt.json",
                file_required=True,
                label="Stage 12 test receipt",
            )
            success_path = confined_relative_path(
                path,
                "_SUCCESS.json",
                file_required=True,
                label="Stage 12 test success receipt",
            )
            receipt, receipt_file = load_json_object(
                receipt_path,
                trusted_root=project_root,
            )
            success, success_file = load_json_object(
                success_path,
                trusted_root=project_root,
            )
        except Stage12InventoryError as exc:
            raise Stage12FreezeError("test checkpoint path is unsafe") from exc
        fingerprint = _as_mapping(
            receipt.get("input_fingerprint"), "test input fingerprint is missing"
        )
        name = str(fingerprint.get("node_name"))
        observed_names.add(name)
        summary = _as_mapping(receipt.get("summary"), "test summary is missing")
        observed_by_name[name] = {
            "fingerprint": fingerprint,
            "checkpoint_id": checkpoint_id,
            "summary": summary,
        }
        access_audit = _as_mapping(
            receipt.get("access_audit"), "test access audit is missing"
        )
        access_counters = _as_mapping(
            access_audit.get("counters"), "test access counters are missing"
        )
        junit = _as_mapping(receipt.get("junit_receipt"), "JUnit receipt is missing")
        if (
            checkpoint_id != receipt.get("checkpoint_id")
            or not checkpoint_id.startswith(f"stage12-checkpoint-{name}-")
            or item.get("receipt_sha256") != receipt_file.sha256
            or item.get("success_sha256") != success_file.sha256
            or success.get("receipt_sha256") != receipt_file.sha256
            or success.get("checkpoint_id") != checkpoint_id
            or receipt.get("schema_version") != 2
            or receipt.get("status") != "PASSED"
            or success.get("status") != "PASSED"
            or receipt.get("input_fingerprint_sha256")
            != sha256_bytes(canonical_json_bytes(fingerprint))
            or fingerprint.get("project_tree_sha256") != project_tree_sha256
            or fingerprint.get("environment_sha256") != environment_sha256
            or fingerprint.get("environment_overrides") != {"PYTHONHASHSEED": "0"}
            or fingerprint.get("process_environment")
            != {**SANITIZED_ENVIRONMENT, "PYTHONHASHSEED": "0"}
            or item.get("summary") != summary
            or item.get("junit_receipt") != junit
            or item.get("access_audit") != access_audit
            or junit.get("test_case_identity_sha256")
            != summary.get("test_case_identity_sha256")
            or not isinstance(junit.get("sha256"), str)
            or len(str(junit.get("sha256"))) != 64
            or int(summary.get("failures", -1)) != 0
            or int(summary.get("errors", -1)) != 0
            or int(summary.get("skipped", -1)) != 0
            or int(summary.get("tests", 0)) <= 0
            or access_audit.get("status") != "NO_FORBIDDEN_PAYLOAD_ACCESS"
            or any(
                int(access_counters.get(counter, -1)) != 0
                for counter in (
                    "forbidden_access_attempts_denied",
                    "process_spawn_attempts_denied",
                    "raw_payload_files_opened",
                    "interim_payload_files_opened",
                    "processed_payload_files_opened",
                    "desktop_payload_files_opened",
                    "final_period_payload_files_opened",
                    "final_period_rows_read",
                )
            )
        ):
            raise Stage12FreezeError(f"test checkpoint differs: {name}")
        normalized.append(
            {
                "checkpoint_id": checkpoint_id,
                "relative_path": relative,
                "receipt_sha256": receipt_file.sha256,
                "success_sha256": success_file.sha256,
                "input_fingerprint_sha256": receipt.get("input_fingerprint_sha256"),
                "status": "PASSED",
                "summary": dict(summary),
                "junit_receipt": dict(junit),
                "access_audit": dict(access_audit),
            }
        )
    if observed_names != expected_names:
        raise Stage12FreezeError("Stage 12 test node names differ")
    test_policy = _as_mapping(policy.get("tests"), "test policy is missing")
    focused_fingerprint = observed_by_name["focused_stage12_protocol"]["fingerprint"]
    independent_fingerprint = observed_by_name["independent_synthetic_regression"][
        "fingerprint"
    ]
    if (
        focused_fingerprint.get("command") != test_policy.get("focused_command")
        or focused_fingerprint.get("dependency_ids") != []
        or independent_fingerprint.get("command")
        != test_policy.get("independent_synthetic_command")
        or independent_fingerprint.get("dependency_ids")
        != [observed_by_name["focused_stage12_protocol"]["checkpoint_id"]]
    ):
        raise Stage12FreezeError("Stage 12 test command/dependency graph differs")
    focused_summary = _as_mapping(
        observed_by_name["focused_stage12_protocol"]["summary"],
        "focused test summary is missing",
    )
    independent_summary = _as_mapping(
        observed_by_name["independent_synthetic_regression"]["summary"],
        "independent test summary is missing",
    )
    if (
        focused_summary.get("tests") != independent_summary.get("tests")
        or focused_summary.get("passed") != independent_summary.get("passed")
        or focused_summary.get("test_case_identity_sha256")
        != independent_summary.get("test_case_identity_sha256")
    ):
        raise Stage12FreezeError(
            "focused and independent synthetic collection identities differ"
        )
    normalized.sort(key=lambda item: item["checkpoint_id"])
    return {
        "schema_version": 1,
        "status": "ALL_REQUIRED_TEST_NODES_PASSED",
        "node_count": len(normalized),
        "nodes": normalized,
        "memoization_rule": (
            "reuse_only_after_exact_code_environment_command_and_dependency_hash_match"
        ),
    }


def _receipt(payload: bytes) -> dict[str, Any]:
    return {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def prepare_stage12_freeze(
    test_receipts: Mapping[str, Any],
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> PreparedFreeze:
    """Recompute every primary document from frozen metadata and test receipts."""

    root = project_root.absolute()
    policy, policy_sha = _load_policy(policy_path, project_root=root)
    methodology = _validate_methodology(policy, project_root=root)
    _, calibration_scope_amendment = _validate_calibration_scope_amendment(
        policy, project_root=root
    )
    _validate_stage12_approval(policy, project_root=root)
    _, post_unblind_technical_amendment = _validate_post_unblind_technical_amendment(
        policy,
        project_root=root,
    )
    _, post_unblind_scratch_topology_amendment = (
        _validate_post_unblind_scratch_topology_amendment(
            policy,
            project_root=root,
        )
    )
    _, post_unblind_zero_row_merge_provenance_amendment = (
        _validate_post_unblind_zero_row_merge_provenance_amendment(
            policy,
            project_root=root,
        )
    )
    _, stage13_source_checkpoint_context_migration = (
        _validate_stage13_source_checkpoint_context_migration(
            policy,
            project_root=root,
        )
    )
    _, december_market_input_scope_partition_amendment = (
        _validate_post_unblind_december_market_input_scope_partition_amendment(
            policy,
            project_root=root,
        )
    )
    _, entry_market_canonicalization_scope_alignment_amendment = (
        _validate_post_unblind_entry_market_canonicalization_scope_alignment_amendment(
            policy,
            project_root=root,
        )
    )
    _, accounting_missing_execution_cost_transport_amendment = (
        _validate_post_unblind_accounting_missing_execution_cost_transport_amendment(
            policy,
            project_root=root,
        )
    )
    _, logical_fingerprint_transport_amendment = (
        _validate_post_unblind_logical_fingerprint_transport_amendment(
            policy,
            project_root=root,
        )
    )
    _, factor_covariance_ordinal_basis_alignment_amendment = (
        _validate_post_unblind_factor_covariance_ordinal_basis_alignment_amendment(
            policy,
            project_root=root,
        )
    )
    _, equal_weight_missing_cost_gate_amendment = (
        _validate_post_unblind_equal_weight_missing_cost_gate_amendment(
            policy,
            project_root=root,
        )
    )
    _, stage13_completion_consolidation = _validate_stage13_completion_consolidation(
        policy,
        project_root=root,
    )
    final_system, final_system_source = _validate_final_system(
        policy, project_root=root
    )
    readiness = _as_mapping(
        final_system.get("implementation_readiness"),
        "implementation readiness is missing",
    )
    if readiness.get("stage13_technical_gate_passed") is not True:
        raise Stage12FreezeError(
            "Stage 12 publication is blocked until the certified final economic "
            "kernel, schema-bound materializer, and exact Arrow schemas exist: "
            f"{readiness.get('blockers')}"
        )
    templates = _validate_report_templates(policy, project_root=root)
    final_input_registry, registry_source = _validate_final_input_registry(
        policy, project_root=root
    )
    project_inventory = build_project_file_inventory(policy, project_root=root)
    replacement_inventory_delta = build_replacement_inventory_delta(
        project_root=root,
        project_inventory=project_inventory,
        policy=policy,
    )
    environment = build_environment_lock(project_root=root)
    environment_sha = sha256_bytes(canonical_json_bytes(environment))
    authorities = authenticate_authority_inventory(policy, project_root=root)
    if authorities.get("authority_count") != 50:
        raise Stage12FreezeError("Stage 12 v12 authority count differs")
    source_drift = build_consolidated_source_drift_record(
        policy=policy,
        project_root=root,
        project_inventory=project_inventory,
    )
    source = _as_mapping(
        policy.get("development_trial_source"), "trial source is missing"
    )
    expected_trial_source = (
        Path.home()
        / "Desktop/Thesis/processed/phase_04_models/phase04-selector-v1"
        / "selector_builds/phase04-selector-results-09a59d61ed58e77d80448a97"
        / "selector_configuration_metrics.parquet"
    ).absolute()
    supplied_trial_source = Path(str(source.get("path"))).absolute()
    if (
        supplied_trial_source != expected_trial_source
        or source.get("sha256")
        != "28bd5def81d7ef738a00e55f085d242668bceb004238a63cd8850ff3ccf3dda4"
        or source.get("rows") != 346
        or source.get("final_period_content") is not False
    ):
        raise Stage12FreezeError("development trial source contract differs")
    try:
        authenticated_trial_source = confined_existing_path(
            supplied_trial_source,
            trusted_root=Path.home() / "Desktop/Thesis",
            file_required=True,
            label="authenticated development trial metadata",
        )
    except Stage12InventoryError as exc:
        raise Stage12FreezeError("development trial source path is unsafe") from exc
    ledger = build_frozen_trial_ledger(
        authenticated_trial_source,
        expected_source_sha256=str(source.get("sha256")),
        expected_source_rows=int(source.get("rows", -1)),
    ).to_payload()
    if ledger.get("raw_trial_count") != EXPECTED_RAW_TRIAL_COUNT:
        raise Stage12FreezeError("frozen raw trial count differs")
    tests = _validate_test_receipts(
        test_receipts,
        policy=policy,
        project_root=root,
        project_tree_sha256=str(project_inventory["tree_sha256"]),
        environment_sha256=environment_sha,
    )
    final_system_document = {
        "schema_version": 1,
        "source_receipt": final_system_source,
        "system": final_system,
    }
    final_input_registry_document = {
        "schema_version": 1,
        "source_receipt": registry_source,
        "registry": final_input_registry,
    }
    checks = {
        "schema_version": 1,
        "status": "ALL_POST_UNBLIND_REPLACEMENT_FREEZE_GATES_PASSED",
        "checks": {
            "methodology_authenticated": True,
            "pre_unblind_calibration_scope_amendment_authenticated": True,
            "post_unblind_technical_amendment_authenticated": True,
            "post_unblind_hash_audit_incident_authenticated": True,
            "post_unblind_adapter_transition_exact": True,
            "post_unblind_amendment_zero_outcome_boundary_authenticated": True,
            "post_unblind_scratch_topology_amendment_authenticated": True,
            "post_unblind_scratch_topology_exact_and_non_mutating": True,
            "post_unblind_scratch_topology_zero_payload_boundary_authenticated": True,
            "post_unblind_zero_row_merge_provenance_amendment_authenticated": True,
            "post_unblind_zero_row_merge_provenance_exact_scope_authenticated": True,
            "post_unblind_zero_row_merge_provenance_non_economic_boundary_authenticated": True,
            "stage13_source_checkpoint_context_migration_authenticated": True,
            "stage13_parent_historical_closed_tree_identity_bound": True,
            "stage13_parent_context_open_still_requires_replacement_approval": True,
            "post_unblind_december_market_input_scope_partition_amendment_authenticated": True,
            "prior_unguarded_empirical_regression_access_disclosed": True,
            "december_exact_entry_market_projection_scope_authenticated": True,
            "decision_source_preflight_and_completeness_gate_authenticated": True,
            "stage13_decision_checkpoint_rebind_authenticated": True,
            "stage13_decision_checkpoint_rebind_431_checkpoint_pins_authenticated": True,
            "stage13_preserved_decision_checkpoint_cache_reauthentication_contract_authenticated": True,
            "stage13_decision_checkpoint_rebind_stage12_synthetic_only": True,
            "checkpoint_rebind_metadata_exception_authenticated": True,
            "checkpoint_rebind_metadata_source_transition_authenticated": True,
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authenticated": True,
            "entry_market_exact_duplicate_canonicalization_contract_authenticated": True,
            "entry_market_closed_tombstone_share_coverage_contract_authenticated": True,
            "entry_market_projection_truthful_receipt_algebra_authenticated": True,
            "entry_market_production_preflight_disclosure_authenticated": True,
            "entry_market_no_third_projection_blocker_authenticated": True,
            "entry_market_standing_stage13_approval_authenticated": True,
            "entry_market_runtime_source_transition_exact": True,
            "post_unblind_accounting_missing_execution_cost_transport_amendment_authenticated": True,
            "accounting_missing_numeric_transport_preserves_nontradable_semantics": True,
            "accounting_frozen_primary_borrow_mapping_authenticated": True,
            "accounting_prewrite_nonnullable_audit_authenticated": True,
            "accounting_transport_non_economic_boundary_authenticated": True,
            "thesis_error_materiality_policy_authenticated": True,
            "post_unblind_logical_fingerprint_transport_amendment_authenticated": True,
            "logical_fingerprint_schema_and_shape_exact": True,
            "logical_fingerprint_null_masks_and_finite_bits_exact": True,
            "logical_fingerprint_nan_payload_canonicalization_only": True,
            "logical_fingerprint_persisted_authentication_unchanged": True,
            "logical_fingerprint_non_economic_boundary_authenticated": True,
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authenticated": True,
            "factor_covariance_zero_based_contract_authenticated": True,
            "factor_covariance_numeric_bits_unchanged": True,
            "factor_covariance_matrix_semantics_unchanged": True,
            "factor_covariance_non_economic_boundary_authenticated": True,
            "post_unblind_equal_weight_missing_cost_gate_amendment_authenticated": True,
            "equal_weight_new_position_execution_gate_authenticated": True,
            "equal_weight_current_risk_input_gate_authenticated": True,
            "equal_weight_required_exit_exclusion_authenticated": True,
            "equal_weight_missing_cost_inherited_exact_hold_authenticated": True,
            "equal_weight_frozen_hold_beta_fallback_boundary_authenticated": True,
            "equal_weight_exact_zero_trade_and_frozen_gross_failure_authenticated": True,
            "equal_weight_residual_gross_allocation_authenticated": True,
            "equal_weight_invalid_input_zero_capacity_and_nonzero_trade_fail_closed_authenticated": True,
            "equal_weight_descriptive_candidate_audit_authenticated": True,
            "equal_weight_rule_restoring_economic_boundary_authenticated": True,
            "v10_model_state_direct_reuse_prohibited": True,
            "fresh_v12_six_model_fits_required": True,
            "stage12_v12_completion_consolidation_authenticated": True,
            "unpublished_v11_recorded_as_non_authoritative_and_superseded": True,
            "substantive_2021_terminal_event_treatment_authenticated": True,
            "terminal_vendor_missing_delisting_return_preserved": True,
            "terminal_primary_cash_retirement_multiplier_one_authenticated": True,
            "terminal_counterfactual_sensitivity_contract_authenticated": True,
            "terminal_sensitivity_implementation_and_focused_tests_bound": True,
            "final_input_registry_1210_column_contract_authenticated": True,
            "historical_stage13_v1_report_template_preserved": True,
            "active_stage13_v2_report_template_authenticated": True,
            "stage13_empirical_result_assembler_v3_authenticated": True,
            "standing_stage13_completion_instruction_hash_authenticated": True,
            "fresh_v12_binding_specific_stage13_approval_required": True,
            "upstream_authority_graph_authenticated": True,
            "stage11_verified_and_not_recomputed": True,
            "complete_project_tree_hashed": True,
            "prior_to_replacement_inventory_delta_exact": True,
            "frozen_models_scores_and_other_strategy_rules_unchanged": True,
            "environment_locked": True,
            "trial_ledger_complete": True,
            "raw_dsr_trial_count_equals_386": True,
            "final_calendar_and_partial_month_rules_frozen": True,
            "benchmarks_metrics_inference_and_robustness_frozen": True,
            "report_templates_frozen": True,
            "final_input_registry_frozen": True,
            "strict_numeric_materiality_and_aggregate_cap_frozen": True,
            "focused_protocol_tests_passed": True,
            "independent_synthetic_regression_passed": True,
            "monthly_stage13_memoization_required": True,
            "stage13_still_requires_separate_user_approval": True,
            "certified_final_economic_kernel_available": True,
            "schema_bound_registration_materializer_available": True,
            "complete_final_input_arrow_schemas_available": True,
            "stage13_technical_gate_passed": True,
            "final_period_access_not_authorized": True,
        },
        "report_template_receipts": templates,
        "methodology_receipts": methodology,
        "calibration_scope_amendment_receipt": calibration_scope_amendment,
        "post_unblind_technical_amendment_receipt": (post_unblind_technical_amendment),
        "post_unblind_hash_audit_incident_receipt": (
            post_unblind_technical_amendment["hash_audit_incident"]
        ),
        "post_unblind_scratch_topology_amendment_receipt": (
            post_unblind_scratch_topology_amendment
        ),
        "post_unblind_zero_row_merge_provenance_amendment_receipt": (
            post_unblind_zero_row_merge_provenance_amendment
        ),
        "stage13_source_checkpoint_context_migration_receipt": (
            stage13_source_checkpoint_context_migration
        ),
        "post_unblind_december_market_input_scope_partition_amendment_receipt": (
            december_market_input_scope_partition_amendment
        ),
        "checkpoint_rebind_metadata_exception_receipt": (
            december_market_input_scope_partition_amendment[
                "checkpoint_rebind_metadata_exception"
            ]
        ),
        "post_unblind_entry_market_canonicalization_scope_alignment_amendment_receipt": (
            entry_market_canonicalization_scope_alignment_amendment
        ),
        "post_unblind_accounting_missing_execution_cost_transport_amendment_receipt": (
            accounting_missing_execution_cost_transport_amendment
        ),
        "post_unblind_logical_fingerprint_transport_amendment_receipt": (
            logical_fingerprint_transport_amendment
        ),
        "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_receipt": (
            factor_covariance_ordinal_basis_alignment_amendment
        ),
        "post_unblind_equal_weight_missing_cost_gate_amendment_receipt": (
            equal_weight_missing_cost_gate_amendment
        ),
        "stage12_v12_completion_consolidation_receipt": (
            stage13_completion_consolidation
        ),
        "counters": {
            "counter_scope": "guarded_stage12_build_and_test_execution_only",
            "guarded_test_processes": 2,
            "authenticated_development_trial_metadata_files_opened": 1,
            "authenticated_development_trial_metadata_rows_read": int(
                source.get("rows", -1)
            ),
            "authenticated_stage11_primary_manifest_files_opened": 1,
            "authenticated_stage11_historical_metadata_files_opened": int(
                source_drift["stage11_historical_metadata_files_opened"]
            ),
            "guarded_test_interim_payload_files_opened": 0,
            "guarded_test_processed_payload_files_opened": 0,
            "guarded_test_desktop_payload_files_opened": 0,
            "raw_payload_files_opened": 0,
            "parent_checkpoint_payload_files_opened": 0,
            "final_period_payload_files_opened": 0,
            "final_period_rows_read": 0,
            "final_forecasts_created": 0,
            "final_weights_created": 0,
            "final_trades_created": 0,
            "final_pnl_or_performance_rows_created": 0,
        },
        "disclosed_amendment_hash_audit": (
            post_unblind_technical_amendment["hash_audit_incident"]["access_facts"]
        ),
        "authorizations": {
            "stage13_technical_gate_passed": True,
            "stage13_requires_separate_user_approval": True,
            "stage13_execution_authorized_now": False,
            "final_period_access_authorized_now": False,
            "post_unblind_technical_amendment_authorized": True,
            "post_unblind_scratch_topology_amendment_authorized": True,
            "post_unblind_zero_row_merge_provenance_amendment_authorized": True,
            "post_unblind_december_market_input_scope_partition_amendment_authorized": True,
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authorized": True,
            "post_unblind_accounting_missing_execution_cost_transport_amendment_authorized": True,
            "post_unblind_logical_fingerprint_transport_amendment_authorized": True,
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authorized": True,
            "post_unblind_equal_weight_missing_cost_gate_amendment_authorized": True,
            "substantive_terminal_event_treatment_frozen": True,
            "standing_stage13_approval_recorded": True,
            "binding_specific_stage13_receipt_may_be_issued_after_v12_verification": True,
            "additional_user_prompt_after_v12_verification_required": False,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_stage13_approval_required": True,
            "renewed_stage13_approval_granted_now": False,
            "stage13_source_checkpoint_context_migration_authorized": True,
            "stage13_decision_checkpoint_rebind_authorized": True,
            "checkpoint_rebind_metadata_correction_authorized": True,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_blockers": list(_STAGE13_BLOCKERS),
        },
    }
    content_documents: dict[str, Mapping[str, Any]] = {
        "authority_inventory.json": authorities,
        "environment_lock.json": environment,
        "file_inventory.json": project_inventory,
        "final_input_registry.json": final_input_registry_document,
        "final_system.json": final_system_document,
        "freeze_checks.json": checks,
        "replacement_inventory_delta.json": replacement_inventory_delta,
        "source_drift.json": source_drift,
        "test_receipts.json": tests,
        "trial_ledger.json": ledger,
    }
    content_bytes = {
        name: canonical_json_bytes(value) for name, value in content_documents.items()
    }
    receipts = {name: _receipt(payload) for name, payload in content_bytes.items()}
    fingerprint_payload = {
        "schema_version": 1,
        "policy_sha256": policy_sha,
        "methodology_source_sha256": methodology["source"]["sha256"],
        "project_tree_sha256": project_inventory["tree_sha256"],
        "environment_sha256": environment_sha,
        "members": receipts,
    }
    fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    freeze_id = f"stage12-freeze-{fingerprint[:24]}"
    manifest = {
        "schema_version": 1,
        "stage": 12,
        "policy_id": policy["policy_id"],
        "freeze_id": freeze_id,
        "content_fingerprint": fingerprint,
        "status": policy["outputs"]["primary_status"],
        "scope": ("CONSOLIDATED_SUBSTANTIVE_ECONOMIC_POST_UNBLIND_REPLACEMENT_FREEZE"),
        "bindings": {
            "policy_sha256": policy_sha,
            "methodology": methodology,
            "final_system_source": final_system_source,
            "final_input_registry_source": registry_source,
            "project_tree_sha256": project_inventory["tree_sha256"],
            "environment_sha256": environment_sha,
            "authority_inventory_sha256": authorities["authority_inventory_sha256"],
            "trial_ledger_sha256": ledger["ledger_sha256"],
            "source_amendment_sha256": source_drift["amendment_receipt"]["sha256"],
            "calibration_scope_amendment_sha256": calibration_scope_amendment["sha256"],
            "base_stage12_policy_sha256": _EXPECTED_BASE_POLICY_SHA256,
            "prior_replacement_policy_sha256": (
                _EXPECTED_PRIOR_V10_REPLACEMENT_POLICY_SHA256
            ),
            "unpublished_superseded_v11_policy_sha256": (
                _EXPECTED_UNPUBLISHED_V11_POLICY_SHA256
            ),
            "stage13_public_interest_last_tradable_close_liquidation_policy_sha256": (
                stage13_completion_consolidation["terminal_treatment_policy"]["sha256"]
            ),
            "stage13_input_registry_total_arrow_column_count": 1210,
            "stage13_report_template_v1_historical_sha256": next(
                item["sha256"]
                for item in templates
                if item["role"] == "historical_superseded_stage13"
            ),
            "stage13_report_template_v2_active_sha256": next(
                item["sha256"] for item in templates if item["role"] == "active_stage13"
            ),
            "stage14_report_template_v1_active_sha256": next(
                item["sha256"] for item in templates if item["role"] == "active_stage14"
            ),
            "stage13_result_assembler_id": ("stage13-empirical-result-assembler-v3"),
            "stage13_completion_standing_instruction_sha256": (
                "13f24431ba741ce24158877302c803cb18a7f8a4ae435e67376594edeef05681"
            ),
            "stage12_v12_completion_consolidation": (stage13_completion_consolidation),
            "post_unblind_equal_weight_missing_cost_gate_amendment_sha256": (
                equal_weight_missing_cost_gate_amendment["sha256"]
            ),
            "post_unblind_equal_weight_missing_cost_gate_amendment_semantic_sha256": (
                _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_SEMANTIC_SHA256
            ),
            "equal_weight_missing_cost_gate_authorization_sha256": (
                _EXPECTED_EQUAL_WEIGHT_MISSING_COST_GATE_AUTHORIZATION_SHA256
            ),
            "stage13_resume3_failure_sha256": (
                _EXPECTED_STAGE13_RESUME3_FAILURE_SHA256
            ),
            "stage13_equal_weight_missing_cost_preflight_sha256": (
                _EXPECTED_STAGE13_EQUAL_WEIGHT_MISSING_COST_PREFLIGHT_SHA256
            ),
            "prior_equal_weight_stage12_file_inventory_sha256": (
                _PRIOR_V10_STAGE12_FILE_INVENTORY_SHA256
            ),
            "prior_equal_weight_stage12_primary_pointer_sha256": (
                "115e2f65e16b1b30bcdaab13bfe9c9c55e0af8d6d98a0b7a3564bab7d8012011"
            ),
            "prior_equal_weight_stage12_verified_pointer_sha256": (
                "f71ef82e9db06df23acf33562dd40f40d426930478706f7c36885d2de7bc4c2b"
            ),
            "prior_equal_weight_stage13_approval_sha256": (
                "23423442f51da1df76bee27e1c93054b710091a38e40bd120b415d6f4a98dfd6"
            ),
            "equal_weight_kernel_pre_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_KERNEL_PRE_SHA256
            ),
            "equal_weight_kernel_post_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_KERNEL_POST_SHA256
            ),
            "equal_weight_publication_pre_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_PUBLICATION_PRE_SHA256
            ),
            "equal_weight_publication_post_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_PUBLICATION_POST_SHA256
            ),
            "equal_weight_runtime_pre_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_RUNTIME_PRE_SHA256
            ),
            "equal_weight_runtime_post_amendment_sha256": (
                _EXPECTED_EQUAL_WEIGHT_RUNTIME_POST_SHA256
            ),
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_sha256": (
                factor_covariance_ordinal_basis_alignment_amendment["sha256"]
            ),
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_semantic_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_SEMANTIC_SHA256
            ),
            "factor_covariance_ordinal_basis_alignment_authorization_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AUTHORIZATION_SHA256
            ),
            "stage13_factor_covariance_ordinal_basis_blocker_sha256": (
                _EXPECTED_STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_BLOCKER_SHA256
            ),
            "stage13_factor_covariance_ordinal_basis_preflight_sha256": (
                _EXPECTED_STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_PREFLIGHT_SHA256
            ),
            "failed_stage13_v9_launch_memo_sha256": (
                _EXPECTED_FAILED_STAGE13_V9_LAUNCH_MEMO_SHA256
            ),
            "prior_factor_covariance_stage12_file_inventory_sha256": (
                _PRIOR_V9_STAGE12_FILE_INVENTORY_SHA256
            ),
            "prior_factor_covariance_stage12_primary_pointer_sha256": (
                "a9799dbf0e033c9ffb0acf771037bb31fd7bd5b50cb4434272d9cd9857755117"
            ),
            "prior_factor_covariance_stage12_verified_pointer_sha256": (
                "69bc5ad1514fefd5a21a1ab9b7318a26ecd89862f3f07e86f8d8ad166bea4ec0"
            ),
            "prior_factor_covariance_stage13_approval_sha256": (
                "ac4432190a5b0a33b8c892f14e13cde0f06bb97b5d4b49878b2e613061202408"
            ),
            "factor_covariance_compiler_pre_amendment_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_COMPILER_PRE_SHA256
            ),
            "factor_covariance_compiler_post_amendment_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_COMPILER_POST_SHA256
            ),
            "factor_covariance_runtime_pre_amendment_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_RUNTIME_PRE_SHA256
            ),
            "factor_covariance_runtime_post_amendment_sha256": (
                _EXPECTED_FACTOR_COVARIANCE_RUNTIME_POST_SHA256
            ),
            "post_unblind_logical_fingerprint_transport_amendment_sha256": (
                logical_fingerprint_transport_amendment["sha256"]
            ),
            "post_unblind_logical_fingerprint_transport_amendment_semantic_sha256": (
                _EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_SEMANTIC_SHA256
            ),
            "logical_fingerprint_transport_authorization_sha256": (
                _EXPECTED_LOGICAL_FINGERPRINT_TRANSPORT_AUTHORIZATION_SHA256
            ),
            "stage13_final_market_input_logical_fingerprint_blocker_sha256": (
                _EXPECTED_STAGE13_LOGICAL_FINGERPRINT_BLOCKER_SHA256
            ),
            "stage13_final_market_input_logical_fingerprint_preflight_sha256": (
                _EXPECTED_STAGE13_LOGICAL_FINGERPRINT_PREFLIGHT_SHA256
            ),
            "prior_logical_fingerprint_stage12_primary_pointer_sha256": (
                "1a2314c978d92222e80d9a625fd2fecb692243dc7a964f5ead530c22bb6e4048"
            ),
            "prior_logical_fingerprint_stage12_verified_pointer_sha256": (
                "979d51cff41b05e3e6e32880ac91ba54f35c196eade63f2fce2483da70c7eb9c"
            ),
            "prior_logical_fingerprint_stage13_approval_sha256": (
                "5bafb96b36c7fc0e02ba44e4c607270a24c5d836c520a037b021506084dd1d4a"
            ),
            "logical_fingerprint_replay_pre_amendment_sha256": (
                _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_PRE_SHA256
            ),
            "logical_fingerprint_replay_post_amendment_sha256": (
                _EXPECTED_LOGICAL_FINGERPRINT_REPLAY_POST_SHA256
            ),
            "post_unblind_accounting_missing_execution_cost_transport_amendment_sha256": (
                accounting_missing_execution_cost_transport_amendment["sha256"]
            ),
            "accounting_missing_execution_cost_transport_authorization_sha256": (
                _EXPECTED_ACCOUNTING_TRANSPORT_AUTHORIZATION_SHA256
            ),
            "thesis_error_materiality_policy_sha256": (
                _EXPECTED_THESIS_ERROR_MATERIALITY_POLICY_SHA256
            ),
            "stage13_adv20_nullability_blocker_sha256": (
                _EXPECTED_STAGE13_ADV20_NULLABILITY_BLOCKER_SHA256
            ),
            "stage13_accounting_missing_execution_cost_preflight_sha256": (
                _EXPECTED_STAGE13_ACCOUNTING_PREFLIGHT_SHA256
            ),
            "prior_accounting_transport_stage12_primary_pointer_sha256": (
                "6f4680f06b3ba773e4ba814122778675e255e60a1d9d22805012112214c2749c"
            ),
            "prior_accounting_transport_stage12_verified_pointer_sha256": (
                "761a7a114ed1aea2feb2bc3b9a0dbec12e003651d2b6c352f8194ac47b2204fc"
            ),
            "prior_accounting_transport_stage13_approval_sha256": (
                "b53bb143fa03ffe93634b07211ad40ca92ff0436b7f7f42abd4c6d04dde16c51"
            ),
            "accounting_transport_replay_pre_amendment_sha256": (
                _EXPECTED_ACCOUNTING_REPLAY_PRE_SHA256
            ),
            "accounting_transport_replay_post_amendment_sha256": (
                _EXPECTED_ACCOUNTING_REPLAY_POST_SHA256
            ),
            "accounting_transport_materializer_pre_amendment_sha256": (
                _EXPECTED_ACCOUNTING_MATERIALIZER_PRE_SHA256
            ),
            "accounting_transport_materializer_post_amendment_sha256": (
                _EXPECTED_ACCOUNTING_MATERIALIZER_POST_SHA256
            ),
            "prior_december_repair_file_inventory_sha256": (
                _PRIOR_V4_STAGE12_FILE_INVENTORY_SHA256
            ),
            "post_unblind_technical_amendment_sha256": (
                post_unblind_technical_amendment["sha256"]
            ),
            "post_unblind_hash_audit_incident_sha256": (
                post_unblind_technical_amendment["hash_audit_incident"]["sha256"]
            ),
            "post_unblind_scratch_topology_amendment_sha256": (
                post_unblind_scratch_topology_amendment["sha256"]
            ),
            "post_unblind_zero_row_merge_provenance_amendment_sha256": (
                post_unblind_zero_row_merge_provenance_amendment["sha256"]
            ),
            "zero_row_merge_provenance_authorization_sha256": (
                _EXPECTED_ZERO_ROW_MERGE_PROVENANCE_AUTHORIZATION_SHA256
            ),
            "stage13_zero_row_copy_guard_blocker_sha256": (
                _EXPECTED_STAGE13_ZERO_ROW_COPY_GUARD_BLOCKER_SHA256
            ),
            "prior_zero_row_provenance_stage12_primary_pointer_sha256": (
                "4c8bbec0ffc8c35319f661dacde9ddd292628a6c3a62598c6b863980801eb0c2"
            ),
            "prior_zero_row_provenance_stage12_verified_pointer_sha256": (
                "5cda903338b0877222991084891afc79bfe277069e81408f1123dc85b93e05b5"
            ),
            "prior_zero_row_provenance_stage13_approval_sha256": (
                "b366335728dda2559b07acbcc4bb7ceb52bdcff64bb6d9993a934425cd1dec4e"
            ),
            "zero_row_adapter_pre_amendment_sha256": (
                _EXPECTED_ZERO_ROW_ADAPTER_PRE_SHA256
            ),
            "zero_row_adapter_post_amendment_sha256": (
                _EXPECTED_ZERO_ROW_ADAPTER_POST_SHA256
            ),
            "zero_row_replay_pre_amendment_sha256": (
                _EXPECTED_ZERO_ROW_REPLAY_PRE_SHA256
            ),
            "zero_row_replay_post_amendment_sha256": (
                _EXPECTED_ZERO_ROW_REPLAY_POST_SHA256
            ),
            "stage13_parent_scratch_topology_blocker_sha256": (
                _EXPECTED_STAGE13_PARENT_SCRATCH_TOPOLOGY_BLOCKER_SHA256
            ),
            "stage13_source_checkpoint_context_migration_v1_sha256": (
                _EXPECTED_STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V1_SHA256
            ),
            "prior_scratch_topology_stage12_primary_pointer_sha256": (
                "04d1105329b09ea078f1abb4c7842ce6441b7c9f70d7869a11bf4954ced274fe"
            ),
            "prior_scratch_topology_stage12_verified_pointer_sha256": (
                "850ba5805753406f9e7002e1cb71f8dfdea51c5d22606c80851baef8929384a6"
            ),
            "prior_scratch_topology_stage13_approval_sha256": (
                "3eece897f35e123659bc2fbf5721b42122d8be0394363d2fe31137c8d53548f2"
            ),
            "pre_amendment_adapter_sha256": (_EXPECTED_PRE_AMENDMENT_ADAPTER_SHA256),
            "post_amendment_adapter_sha256": (_EXPECTED_POST_AMENDMENT_ADAPTER_SHA256),
            "prior_stage12_primary_pointer_sha256": (
                "7ebd59fe8d803cd7da06b0da8a1400f061bf043bf196f37d4a5639f0b5ca5735"
            ),
            "prior_stage12_verified_pointer_sha256": (
                "75c45ed9d6c3c6f326d8d148224c3036fb1f331f734bf3ae3d4245c7be5df75e"
            ),
            "prior_stage13_approval_sha256": (
                "dab8e1465dc95eb46323ada79c7978151317fc55eacd35d95c3ae2325667049d"
            ),
            "stage13_resume_blocker_sha256": (
                "d6c212f5272adcf2396f64a4483ce74de3ae74a0eed17c110ab569fc03c852a1"
            ),
            "replacement_inventory_delta_sha256": receipts[
                "replacement_inventory_delta.json"
            ]["sha256"],
            "stage13_source_checkpoint_context_migration_sha256": (
                stage13_source_checkpoint_context_migration["sha256"]
            ),
            "stage13_parent_feature_sources_tree_sha256": (
                stage13_source_checkpoint_context_migration["parent_feature_sources"][
                    "canonical_tree_sha256"
                ]
            ),
            "post_unblind_december_market_input_scope_partition_amendment_sha256": (
                december_market_input_scope_partition_amendment["sha256"]
            ),
            "december_market_input_scope_partition_authorization_sha256": (
                december_market_input_scope_partition_amendment[
                    "authorization_receipt"
                ]["sha256"]
            ),
            "stage13_december_market_input_blocker_sha256": (
                december_market_input_scope_partition_amendment["blocker_receipt"][
                    "sha256"
                ]
            ),
            "prior_december_repair_stage12_primary_pointer_sha256": (
                "72f1f01a00ea2789db5131172f1a8609da4295a00b27ee919b09528fca60debf"
            ),
            "prior_december_repair_stage12_verified_pointer_sha256": (
                "a333c7026c014a4a030a9a8f3548fb0fc2516f5786f4ec1023a9755a93db2c74"
            ),
            "prior_december_repair_stage13_approval_sha256": (
                "2bc69b11d224f7915d0c63687daf1fa4f6212e97d725ed3602c6f8ef940277ea"
            ),
            "stage13_decision_checkpoint_rebind_sha256": (
                december_market_input_scope_partition_amendment[
                    "decision_checkpoint_rebind"
                ]["sha256"]
            ),
            "stage13_decision_checkpoint_rebind_v1_sha256": (
                _EXPECTED_STAGE13_DECISION_CHECKPOINT_REBIND_V1_SHA256
            ),
            "checkpoint_rebind_metadata_correction_authorization_sha256": (
                _EXPECTED_CHECKPOINT_REBIND_METADATA_AUTHORIZATION_SHA256
            ),
            "stage13_checkpoint_rebind_metadata_blocker_sha256": (
                _EXPECTED_CHECKPOINT_REBIND_METADATA_BLOCKER_SHA256
            ),
            "stage13_checkpoint_rebind_metadata_preflight_sha256": (
                _EXPECTED_CHECKPOINT_REBIND_METADATA_PREFLIGHT_SHA256
            ),
            "prior_checkpoint_rebind_metadata_stage12_primary_pointer_sha256": (
                "81eb2bc85d960e9a005f6b62b64a8d8444df75068685ad90cd8c58f49c7d5bdf"
            ),
            "prior_checkpoint_rebind_metadata_stage12_manifest_sha256": (
                "5599c0efdd7d1a334071314ba8e68022a54701039419350c9b19d6b150984a97"
            ),
            "prior_checkpoint_rebind_metadata_stage12_verified_pointer_sha256": (
                "3a19790a88a64542f9b42429487e4b892d905e88fc1142ce100974f344720fca"
            ),
            "prior_checkpoint_rebind_metadata_verification_report_sha256": (
                "c60922692c2fd12bcea80975fbe1ee72496b5a82c37608f697002e8cba0853e3"
            ),
            "prior_checkpoint_rebind_metadata_stage13_approval_sha256": (
                "4286d671963429bfe5c113663ca37decdaffa1f34b95cad54b5f481874dbcd49"
            ),
            "prior_checkpoint_rebind_metadata_stage13_approval_payload_sha256": (
                "fc92568e651f79a74b03f7fae4ed79919561b19521ac913d1058cbd4f84daab5"
            ),
            "prior_checkpoint_rebind_metadata_stage12_binding_sha256": (
                "0a95ac10ff1103a788bb4f60f5a4321aa483dceec0c8bcebb8670da7adcc668c"
            ),
            "checkpoint_rebind_final_source_replay_sha256": (
                _EXPECTED_CHECKPOINT_REBIND_REPLAY_POST_SHA256
            ),
            "stage13_decision_checkpoint_rebind_receipt_count": 431,
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_sha256": (
                entry_market_canonicalization_scope_alignment_amendment["sha256"]
            ),
            "entry_market_canonicalization_scope_alignment_authorization_sha256": (
                entry_market_canonicalization_scope_alignment_amendment[
                    "authorization_receipt"
                ]["sha256"]
            ),
            "stage13_entry_market_projection_contract_blocker_sha256": (
                entry_market_canonicalization_scope_alignment_amendment[
                    "combined_projection_blocker_receipt"
                ]["sha256"]
            ),
            "stage13_entry_market_share_coverage_preflight_sha256": (
                entry_market_canonicalization_scope_alignment_amendment[
                    "share_coverage_preflight_receipt"
                ]["sha256"]
            ),
            "prior_entry_market_alignment_file_inventory_sha256": (
                _PRIOR_V5_STAGE12_FILE_INVENTORY_SHA256
            ),
            "prior_entry_market_alignment_policy_sha256": (
                _EXPECTED_PRIOR_V5_REPLACEMENT_POLICY_SHA256
            ),
            "prior_entry_market_alignment_stage12_primary_pointer_sha256": (
                "5d8820eba667dee1f311811ad7e19224fa5b2d67078224df258765012d5c00fe"
            ),
            "prior_entry_market_alignment_stage12_verified_pointer_sha256": (
                "aed589c5d3740337edc8f4298abc72ec0f9971fc0c0cb3aac289b61abd3b4028"
            ),
            "prior_entry_market_alignment_stage13_approval_sha256": (
                "dc856b1954e7d7779a43e7c0b7ecf0102f9a7fd8121f2bd3594d6567543283fa"
            ),
            "entry_market_final_source_replay_pre_amendment_sha256": (
                _EXPECTED_ENTRY_MARKET_REPLAY_PRE_SHA256
            ),
            "entry_market_final_source_replay_post_amendment_sha256": (
                _EXPECTED_ENTRY_MARKET_REPLAY_POST_SHA256
            ),
            "entry_market_validation_market_extension_pre_amendment_sha256": (
                _EXPECTED_ENTRY_MARKET_EXTENSION_PRE_SHA256
            ),
            "entry_market_validation_market_extension_post_amendment_sha256": (
                _EXPECTED_ENTRY_MARKET_EXTENSION_POST_SHA256
            ),
        },
        "members": receipts,
        "counters": checks["counters"],
        "authorizations": checks["authorizations"],
        "memoization": {
            "completed_stage11_replayed": False,
            "test_checkpoint_ids": [item["checkpoint_id"] for item in tests["nodes"]],
            "stage13_monthly_content_addressed_checkpoints_required": True,
            "resume_from_latest_authenticated_month": True,
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    documents = {
        **content_documents,
        "pre_final_manifest.json": manifest,
    }
    document_bytes = {**content_bytes, "pre_final_manifest.json": manifest_bytes}
    document_receipts = {
        **receipts,
        "pre_final_manifest.json": _receipt(manifest_bytes),
    }
    success = {
        "schema_version": 1,
        "stage": 12,
        "freeze_id": freeze_id,
        "content_fingerprint": fingerprint,
        "status": policy["outputs"]["primary_status"],
        "members": dict(document_receipts),
    }
    success_bytes = canonical_json_bytes(success)
    try:
        decoded_success = json.loads(success_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12FreezeError("primary success serialization failed") from exc
    if decoded_success != success or success["members"] != document_receipts:
        raise Stage12FreezeError("primary success in-memory/byte identity differs")
    documents["_SUCCESS.json"] = success
    document_bytes["_SUCCESS.json"] = success_bytes
    document_receipts["_SUCCESS.json"] = _receipt(success_bytes)
    if set(document_bytes) != set(PRIMARY_FILES):
        raise Stage12FreezeError("primary closed-member set differs")
    return PreparedFreeze(
        policy=policy,
        policy_sha256=policy_sha,
        policy_path=confined_existing_path(
            policy_path,
            trusted_root=root,
            file_required=True,
            label="Stage 12 freeze policy",
        ),
        documents=documents,
        document_bytes=document_bytes,
        document_receipts=document_receipts,
        freeze_id=freeze_id,
        content_fingerprint=fingerprint,
    )


def _write_member(
    stage: SecureStagingDirectory,
    name: str,
    payload: bytes,
) -> None:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise Stage12FreezeError(f"short Stage 12 bundle write: {name}")


def _authenticate_published(
    prepared: PreparedFreeze,
    *,
    bundle_path: Path,
    trusted_root: Path,
) -> None:
    try:
        bundle = confined_existing_path(
            bundle_path,
            trusted_root=trusted_root,
            file_required=False,
            label="published Stage 12 bundle",
        )
    except (IndexError, Stage12InventoryError) as exc:
        raise Stage12FreezeError(
            "published Stage 12 bundle is not a confined real directory"
        ) from exc
    names = {path.name for path in bundle.iterdir()}
    if names != set(PRIMARY_FILES):
        raise Stage12FreezeError("published Stage 12 closed set differs")
    for name, payload in prepared.document_bytes.items():
        try:
            member = confined_relative_path(
                bundle,
                name,
                file_required=True,
                label="published Stage 12 member",
            )
            record = stable_file(member, trusted_root=bundle)
        except Stage12InventoryError as exc:
            raise Stage12FreezeError(
                f"published Stage 12 member path is unsafe: {name}"
            ) from exc
        if record.sha256 != sha256_bytes(payload) or record.size_bytes != len(payload):
            raise Stage12FreezeError(f"published Stage 12 member differs: {name}")


def publish_stage12_freeze(
    prepared: PreparedFreeze,
    *,
    project_root: Path = PROJECT_ROOT,
) -> PrimaryFreezePublication:
    """Publish the immutable primary bundle, then replace its pointer last."""

    root = project_root.absolute()
    bound_policy, bound_policy_sha = _load_policy(
        prepared.policy_path,
        project_root=root,
    )
    if bound_policy_sha != prepared.policy_sha256:
        raise Stage12FreezeError("prepared Stage 12 policy changed before publication")
    if canonical_json_bytes(prepared.policy) != canonical_json_bytes(bound_policy):
        raise Stage12FreezeError("prepared Stage 12 policy semantics differ")
    outputs = _as_mapping(prepared.policy.get("outputs"), "outputs are missing")
    destination = (
        root
        / str(outputs["artifact_root"])
        / str(outputs["primary_namespace"])
        / prepared.freeze_id
    )
    expected_hashes = {
        name: str(receipt["sha256"])
        for name, receipt in prepared.document_receipts.items()
    }

    def gate() -> None:
        replay = prepare_stage12_freeze(
            prepared.documents["test_receipts.json"],
            policy_path=prepared.policy_path,
            project_root=root,
        )
        if (
            replay.freeze_id != prepared.freeze_id
            or replay.policy_sha256 != prepared.policy_sha256
            or replay.document_bytes != prepared.document_bytes
        ):
            raise Stage12FreezeError("Stage 12 publication gate replay differs")

    with SecureStagingDirectory.create(
        trusted_root=root,
        destination=destination,
        expected_files=PRIMARY_FILES,
        label="Stage 12 primary freeze",
    ) as stage:
        for name in sorted(PRIMARY_FILES):
            _write_member(stage, name, prepared.document_bytes[name])
        snapshot = stage.snapshot(expected_hashes=expected_hashes)
        published = stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=gate,
            post_publish_gate=gate,
        )
    _authenticate_published(
        prepared,
        bundle_path=published.path,
        trusted_root=root,
    )
    pointer = {
        "schema_version": 1,
        "stage": 12,
        "freeze_id": prepared.freeze_id,
        "status": prepared.policy["outputs"]["primary_status"],
        "bundle_path": published.path.relative_to(root).as_posix(),
        "content_fingerprint": prepared.content_fingerprint,
        "manifest_sha256": prepared.document_receipts["pre_final_manifest.json"][
            "sha256"
        ],
        "success_sha256": prepared.document_receipts["_SUCCESS.json"]["sha256"],
        "closed_member_count": len(PRIMARY_FILES),
        "authorizations": {
            "independent_stage12_verification_authorized": True,
            "stage13_technical_gate_passed": True,
            "stage13_requires_separate_user_approval": True,
            "stage13_execution_authorized_now": False,
            "final_period_access_authorized_now": False,
            "post_unblind_technical_amendment_authorized": True,
            "post_unblind_scratch_topology_amendment_authorized": True,
            "post_unblind_zero_row_merge_provenance_amendment_authorized": True,
            "post_unblind_december_market_input_scope_partition_amendment_authorized": True,
            "post_unblind_entry_market_canonicalization_scope_alignment_amendment_authorized": True,
            "post_unblind_accounting_missing_execution_cost_transport_amendment_authorized": True,
            "post_unblind_logical_fingerprint_transport_amendment_authorized": True,
            "post_unblind_factor_covariance_ordinal_basis_alignment_amendment_authorized": True,
            "post_unblind_equal_weight_missing_cost_gate_amendment_authorized": True,
            "substantive_terminal_event_treatment_frozen": True,
            "standing_stage13_approval_recorded": True,
            "binding_specific_stage13_receipt_may_be_issued_after_v12_verification": True,
            "additional_user_prompt_after_v12_verification_required": False,
            "prior_stage13_approval_valid_for_replacement_identity": False,
            "renewed_stage13_approval_required": True,
            "renewed_stage13_approval_granted_now": False,
            "stage13_source_checkpoint_context_migration_authorized": True,
            "stage13_decision_checkpoint_rebind_authorized": True,
            "checkpoint_rebind_metadata_correction_authorized": True,
            "stage13_parent_context_open_authorized_now": False,
            "stage13_blockers": list(_STAGE13_BLOCKERS),
        },
    }
    pointer_bytes = canonical_json_bytes(pointer)
    pointer_path = root / str(outputs["latest_primary_pointer"])
    replace_pointer_last(
        pointer_path,
        pointer_bytes,
        trusted_root=root,
        pre_replace_gate=lambda: _authenticate_published(
            prepared,
            bundle_path=published.path,
            trusted_root=root,
        ),
        label="Stage 12 primary authority pointer",
    )
    pointer_record = stable_file(pointer_path, trusted_root=root)
    if pointer_record.sha256 != sha256_bytes(pointer_bytes):
        raise Stage12FreezeError("Stage 12 primary pointer differs")
    return PrimaryFreezePublication(
        freeze_id=prepared.freeze_id,
        bundle=published,
        pointer_path=pointer_path,
        pointer_sha256=pointer_record.sha256,
        manifest_sha256=str(
            prepared.document_receipts["pre_final_manifest.json"]["sha256"]
        ),
    )


def test_receipt_document(
    nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {"schema_version": 1, "nodes": [dict(item) for item in nodes]}


__all__ = [
    "BASE_POLICY",
    "CALIBRATION_SCOPE_AMENDMENT",
    "DEFAULT_POLICY",
    "MATERIALITY_AMENDMENT",
    "POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT",
    "POST_UNBLIND_TECHNICAL_AMENDMENT",
    "POST_UNBLIND_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT",
    "POST_UNBLIND_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT",
    "POST_UNBLIND_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT",
    "POST_UNBLIND_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT",
    "POST_UNBLIND_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT",
    "POST_UNBLIND_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT",
    "POST_UNBLIND_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT",
    "STAGE13_PUBLIC_INTEREST_LAST_TRADABLE_CLOSE_LIQUIDATION",
    "STAGE13_DECISION_CHECKPOINT_REBIND",
    "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION",
    "PRIMARY_FILES",
    "PROJECT_ROOT",
    "PreparedFreeze",
    "PrimaryFreezePublication",
    "Stage12FreezeError",
    "_validate_post_unblind_december_market_input_scope_partition_amendment",
    "_validate_post_unblind_entry_market_canonicalization_scope_alignment_amendment",
    "_validate_post_unblind_accounting_missing_execution_cost_transport_amendment",
    "_validate_post_unblind_logical_fingerprint_transport_amendment",
    "_validate_post_unblind_factor_covariance_ordinal_basis_alignment_amendment",
    "_validate_post_unblind_equal_weight_missing_cost_gate_amendment",
    "_validate_stage13_completion_consolidation",
    "build_consolidated_source_drift_record",
    "build_source_drift_record",
    "prepare_stage12_freeze",
    "publish_stage12_freeze",
    "test_receipt_document",
]
