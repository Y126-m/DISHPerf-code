# HyPerf四预算扩展：独立版本v5

预算为每工作负载0.5p、p、2p、3p，五次配对。模型仍是DISHPerf-NP和原版HyPerf-PP，筛选最小预算仍为0.5p，原120候选、原TensorFlow模型、HyPerf采样器均不修改。

## 划分与复用

以v4同一数据、同一环境种子重建原2p训练池及各旧预算训练行，顺序保持一致。用固定的种子异或20261026从旧测试池随机预留额外p行（不读取误差来选行），与旧训练池组成3p池。四预算的共同测试集均排除这额外p行。保留旧预算的训练行，扩大最大训练池，而不是重新洗牌全部样本。

reuse阶段验证源包全部冻结哈希、源完成索引和文件哈希、原测试预测误差、训练行顺序和测试子集关系，随后在共同测试集重新评价。新结果明确记录evaluation_reused和reuse_provenance.json，不宣称重新训练、不计作新重复。原始v4不修改。

新测试集是事后增加预算后确定的共同子集，不是一次完全独立、从未查看过的外部验证。此前已查看v4结果，论文应披露本次预算扩展为修订阶段的补充分析。不得将新旧测试口径混在同一学习曲线。

## 本地执行

将完整文件夹放在D:/aresearch/DISHPerf_HyPerf_budget3p_v5。无需新装环境，继续使用现有现代Python及moxiangev。源包默认D:/aresearch/DISHPerf_HyPerf_original_v4。

1. powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage check
2. powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage reuse -System x264 -Budget 0 -RunCount 5 -Model all
3. powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage plan -System x264 -Budget 75 -RunCount 5 -Model all
4. powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage run -System x264 -Budget 75 -StartRep 0 -RunCount 1 -Model all
5. powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage run -System x264 -Budget 75 -RunCount 5 -Model all
6. powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage plot -System x264 -Budget 0 -RunCount 5 -Model all

reuse不训练：x264应复用30个已完成外层任务；新的B75有10个任务待训练。B75第0对正常完成后，五次命令会核验并跳过已完成任务。仍为单进程，适合本地16GB内存。不要同时启动其他本地训练队列。B75完整DISHPerf包含9*120候选拟合加9次重训，验收并不等于快速smoke。

lrzip B33及h2 B48也已登记，但暂不分配到本地，避免与服务器重复。运行器不会读取服务器数据，必须通过-ReuseRoot明确指向已核验的完整旧套件才能复用服务器结果；仅汇总ZIP不足以通过完整验证。

## 统计和结果

Stage plot -System x264 -Budget 0生成四预算共同测试口径的图，未完成预算保留缺口。误差条是样本SD。警告结果保留，不剔除最大值、不换种子。Holm仅在全部3*4=12设置、每设置5对都完成时执行。五对的双侧精确检验不可能得到p<0.05。

复用结果标为timing_resumed=true，记录的历史计算时间不作为本次端到端耗时。源包运行平台及版本通过provenance保留。不要与另一个服务器版本的任务直接改manifest拼接。

旧配置、旧结果、环境目录不会被本脚本修改。run阶段仅向当前v5目录写入。此交付是Windows本地包，不要覆盖服务器原迁移套件。
