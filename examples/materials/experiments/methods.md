# 探索性实验比较方法

方法代号 experiment_comparison_v1。本资料是合成示例，给出的是可讨论的方法而不是已执行的计算。

候选步骤：先列出 run_name、temperature_c、loss、epoch_count 和 approved，标出缺失 loss 的记录；再向用户确认评估条件、重复运行和缺失处理。若条件可比，可提议计算各 variant 相对 baseline 的 loss 差值，并同时呈现 epoch_count，避免忽略训练资源差异。

目前每个 run_name 只有一条记录。没有重复次数、方差或随机种子资料时，不输出显著性结论，也不推断 temperature_c 的独立影响。若用户需要可靠的方案选择，应先补充重复试验计划和一致的评估协议。
