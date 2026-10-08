# ChoiceForge documentation

This page is the entry point for the public documentation. Most experiment
reports are kept in Chinese for now; the English pages below cover the concepts
needed to reproduce the main route.

## Start here

1. [Research direction](RESEARCH.md) / [研究主线](RESEARCH.zh-CN.md)
2. [Algorithm and output contract](ALGORITHM.md) / [输出契约](OUTPUT_CONTRACT.zh-CN.md)
3. [Data and provenance](DATA.md) / [发布数据审计](RELEASE_DATA_AUDIT.zh-CN.md)
4. [Local model bundle](LOCAL_MODEL_BUNDLE_V1.zh-CN.md)
5. [ChoiceForge Skill workflow](../skills/choiceforge/references/workflow.md)

## Evidence path

- [Findings](FINDINGS.zh-CN.md): consolidated conclusions and limitations.
- [Results](RESULTS.zh-CN.md): first public-data calibration study.
- [Encoder/decoder comparison](ENCODER_DECODER_RESULTS.zh-CN.md): frozen route
  selection under the matched public protocol.
- [Typed Decisions objective screen](TYPED_DECISIONS_OBJECTIVE_SCREEN.zh-CN.md):
  representation-frozen objective comparison.
- [Outcome feedback design](EXECUTABLE_OUTCOME_RLCD_DESIGN.zh-CN.md):
  nondifferentiable result-feedback decomposition.
- `evidence/`: machine-readable hashes, metrics and audit outputs.

## Reproduction conventions

- Public source revisions, licenses and transformations are recorded before data
  is used.
- Validation is for development; sealed test data is reserved for locked reports.
- Every run records configuration, seed, selected IDs and artifact hashes.
- Negative results remain part of the evidence chain.

The root [README](../README.md) contains installation, inference and training
commands. The Chinese README is the most complete user guide.
