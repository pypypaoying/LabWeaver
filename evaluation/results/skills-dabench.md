# Skills 外部基准配对评测

## 结果概览

批次完整：24道题、24份不同CSV路径、23种唯一文件内容，每题每组3次；计划144次，已记录144次。

| 指标 | Skills OFF | Skills ON |
|---|---:|---:|
| 官方参考答案匹配（盲整理后） | 64/72（88.89%） | 67/72（93.06%） |
| 产品交付完成 | 64/72（88.89%） | 66/72（91.67%） |
| 参考答案匹配且交付完成 | 61/72（84.72%） | 64/72（88.89%） |

端到端配对差值 ON−OFF：+4.17个百分点；CSV整簇 bootstrap 95%探索性区间 [-6.94, +15.28] 个百分点。

主比较采用 `skills-dabench-v4`。按预先声明的规则采用首个完整且未发生环境中断的批次，不按得分选择最佳批次；所有中断批次保留并单列成本。主表保留冻结评分流程的原成绩，表示修正另列。

区间覆盖零差值；当前样本不足以确认稳定收益。结论仅针对该固定子集，不能据此断定Skills普遍有效或无效。

## 方法与评分

使用官方公开CSV、原始英文问题、constraints及format；按不同CSV文件划分开发与留出集，正式题目在运行前冻结。难度分布：easy 8题、medium 8题、hard 8题。

两组使用相同代码、模型、输入、沙箱与预算；每次创建新进程和会话，配对顺序交错。ON整体包含技能目录、读取引导与按需正文，不能把差值仅归因于正文。

Agent和盲整理器均为 `deepseek-flash`，配置固定。整理器没有数据、工具和标准答案，只可转写原回答已陈述的答案；未陈述内容不得补算。数值来源过滤不能完全证明字段映射和语义转写正确。

官方比较器使用字符串严格相等，或数值绝对差小于 `1e-6`。原评分器会跳过空回答；本轮为全部计划试次补齐失败记录，空答、超时、崩溃与未完成均保留在分母中。

ABQ为整题全部子答案正确率；PASQ为逐题子答案正确比例的均值；UASQ为全部子答案的总体正确率。三者按官方函数计算，保留原始回答分与盲整理后分。

| 官方指标 | OFF 原始 | ON 原始 | OFF 盲整理 | ON 盲整理 |
|---|---:|---:|---:|---:|
| ABQ | 2.78% | 2.78% | 88.89% | 93.06% |
| PASQ | 2.78% | 2.78% | 89.58% | 93.06% |
| UASQ | 2.70% | 1.80% | 90.99% | 94.59% |

开发题仅用于回答格式适配，没有调整Agent提示词。CSV级划分不能保证底层数据家族完全隔离；以下区间也不能排除同源相关性或预训练污染。

已核对原CSV表头：开发集 `test_ave.csv` 与留出集 `test_x.csv`、`titanic.csv`、`titanic_train.csv` 属于Titanic同源或派生数据。因此这些文件不能视作完全独立的数据家族。题目、冻结计划和全部成绩均保留，没有为消除此局限补跑新题。

字节审计还确认 `titanic.csv`、`titanic_train.csv` 的SHA-256完全相同；不同文件路径并不代表不同内容。重复内容对应的题目仍保留在全部原始成绩中。

仅因盲整理从原始未通过变为通过：OFF 62次，ON 65次。整理不作为Agent内部修复，也不修改原始报告。

## 配对与重复稳定性

| 维度 | 双方通过 | 仅OFF通过 | 仅ON通过 | 双方失败 | 缺失配对 |
|---|---:|---:|---:|---:|---:|
| 答案 | 62 | 2 | 5 | 3 | 0 |
| 交付 | 62 | 2 | 4 | 4 | 0 |
| 端到端 | 57 | 4 | 7 | 4 | 0 |

| 维度 | ON−OFF（百分点） | 95%探索性区间（百分点） | CSV簇 | 完整配对／排除配对 |
|---|---:|---|---:|---|
| answer_correct | +4.17 | [-4.17, +12.50] | 24 | 72／0 |
| delivery_completed | +2.78 | [-2.78, +9.72] | 24 | 72／0 |
| end_to_end | +4.17 | [-6.94, +15.28] | 24 | 72／0 |

整簇抽样将同一CSV的全部重复一起抽取；区间仅使用完整OFF/ON配对，缺失数量如上。重复次数不能当成同等数量的独立问题；不同CSV之间也可能共享数据家族，此区间未消除该相关性。

| 全部重复均通过的题目 | OFF | ON |
|---|---:|---:|
| answer_correct | 19/24 | 21/24 |
| delivery_completed | 19/24 | 21/24 |
| end_to_end | 18/24 | 20/24 |

## 效率与实际技能读取

| 指标 | OFF | ON |
|---|---:|---:|
| Agent已知token | 3,284,813 | 4,857,613 |
| Agent用量未知请求 | 0 | 0 |
| Agent未取得计量的试次 | 0 | 0 |
| 额外盲整理调用 | 67 | 69 |
| 盲整理已知token | 54,912 | 57,038 |
| 盲整理用量未知请求 | 0 | 0 |
| 盲整理状态未取得的试次 | 0 | 0 |
| Agent耗时中位数（秒） | 21.52 | 23.49 |
| 代码执行尝试 | 93 | 114 |
| 无需内部修正即端到端完成 | 0 | 1 |
| 实际技能读取 | 0 | 76 |
| 发生读取的试次 | 0 | 59 |
| 经内部反馈或修正后端到端完成 | 61 | 63 |

Token仅汇总接口实际返回的用量；未知不填零、不估算价格。耗时是本机实际运行记录，不是隔离的性能基准。

| 技能 | OFF读取 | ON读取 |
|---|---:|---:|
| data-exploration | 0 | 6 |
| statistical-analysis | 0 | 67 |
| data-visualization | 0 | 1 |
| analysis-validation | 0 | 2 |

## 失败诊断

以下标签来自状态与评分字段，可重叠，数量不能相加作为失败总数。‘答案不符’不直接等同于计算代码错误；题意、方法选择、标准答案和语义转写都可能影响得分。

| 诊断标签 | OFF | ON |
|---|---:|---:|
| 运行协议或交付失败 | 5 | 5 |
| 已交付但官方答案不符 | 3 | 2 |
| 等待用户澄清 | 3 | 1 |
| 未形成可评分答案／整理失败 | 0 | 0 |
| 工作进程或基础设施异常 | 0 | 0 |
| 尚未归因 | 0 | 0 |

## 逐题结果

每项为通过次数／已记录次数。原始CSV、题目答案、聊天、代码、运行路径和日志不随此报告发布。

| 题目 | CSV | 难度／概念 | OFF答案／交付／端到端 | ON答案／交付／端到端 | OFF诊断 | ON诊断 |
|---|---|---|---|---|---|---|
| DA11 | GODREJIND.csv | medium；Correlation Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA218 | ferret-Pitt-2-preinf-lib2-100_sitediffsel.csv | easy；Correlation Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA250 | baseball_data.csv | medium；Feature Engineering, Summary Statistics | 2/3／2/3／2/3 | 3/3／3/3／3/3 | 运行协议或交付失败 1 | — |
| DA271 | 3901.csv | hard；Comprehensive Data Preprocessing | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA33 | imp.score.ldlr.metabolome.csv | easy；Distribution Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA337 | Zip_MedianSoldPricePerSqft_AllHomes.csv | medium；Distribution Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA351 | test_x.csv | easy；Correlation Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA412 | titanic_train.csv | medium；Feature Engineering | 2/3／2/3／1/3 | 3/3／3/3／3/3 | 运行协议或交付失败 1；已交付但官方答案不符 1 | — |
| DA419 | bitconnect_price.csv | hard；Summary Statistics, Distribution Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA427 | cost_data_with_errors.csv | easy；Comprehensive Data Preprocessing | 3/3／3/3／3/3 | 1/3／3/3／1/3 | — | 已交付但官方答案不符 2 |
| DA452 | baro_2015.csv | hard；Correlation Analysis, Feature Engineering | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA480 | oecd_education_spending.csv | hard；Feature Engineering, Summary Statistics | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA496 | percent-bachelors-degrees-women-usa.csv | hard；Feature Engineering, Summary Statistics | 2/3／1/3／1/3 | 2/3／0/3／0/3 | 运行协议或交付失败 2 | 运行协议或交付失败 3 |
| DA518 | titanic.csv | medium；Outlier Detection | 3/3／3/3／3/3 | 3/3／2/3／2/3 | — | 运行协议或交付失败 1 |
| DA542 | abalone.csv | easy；Summary Statistics | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA555 | tree.csv | easy；Feature Engineering | 3/3／2/3／2/3 | 3/3／3/3／3/3 | 运行协议或交付失败 1 | — |
| DA578 | e5_aapl.csv | easy；Summary Statistics | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA59 | estimated_numbers.csv | easy；Distribution Analysis, Summary Statistics, Feature Engineering | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA590 | 20170413_000000_group_statistics.csv | hard；Machine Learning | 0/3／0/3／0/3 | 1/3／1/3／1/3 | 等待用户澄清 3 | 运行协议或交付失败 1；等待用户澄清 1 |
| DA644 | random_stock_data.csv | medium；Distribution Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA673 | my_test_01.csv | hard；Comprehensive Data Preprocessing, Feature Engineering, Correlation Analysis, Summary Statistics | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA716 | Current_Logan.csv | medium；Summary Statistics, Comprehensive Data Preprocessing | 1/3／3/3／1/3 | 3/3／3/3／3/3 | 已交付但官方答案不符 2 | — |
| DA721 | auto-mpg.csv | medium；Correlation Analysis | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |
| DA77 | microsoft.csv | hard；Comprehensive Data Preprocessing, Summary Statistics | 3/3／3/3／3/3 | 3/3／3/3／3/3 | — | — |

## 分组相关性的敏感性分析

以下仅对前文主表原评分调整重采样分组，不改变任何题目、通过次数、主表原CSV簇区间或模型执行。分组规则只依据CSV字节及表头审计，未使用标准答案或模型成绩。两个附加区间仍为探索性估计，已识别家族并非穷尽审计。

| 重采样分组 | 簇数 | 端到端ON−OFF（百分点） | 95%探索性区间（百分点） |
|---|---:|---:|---|
| 完全相同字节合簇 | 23 | +4.17 | [-5.80, +13.35] |
| 相同字节＋已识别家族合簇 | 22 | +4.17 | [-5.56, +13.64] |

相同内容与已知家族合簇能够体现部分相关性，但不能证明其他CSV彼此独立，也不能排除训练污染；不将这些附表当作显著性结论。表示修正版v3在下节单列，未以敏感性区间替换其原CSV簇区间。

## 表示修正版 v3：全样本无模型重评分

开发题暴露了两项回答传输问题：原回答复述公共format占位符时被当成最终值保留，以及Unicode减号不被数值解析识别。v3对所有计划样本使用同一纯表示规则，只移除与原题format完全相同的占位值、统一减号，并重放已有整理器回答。真实原答案（包括错误数值）仍保留；不新增模型调用，不重新计算或选择性重跑任务。

这是开发题导出的事后适配器修正，不能解释为Agent分析能力提升；原始评分、summary及前文主表均保留。

| 官方指标 | OFF 原盲整理 | ON 原盲整理 | OFF 表示v3 | ON 表示v3 |
|---|---:|---:|---:|---:|
| ABQ | 88.89% | 93.06% | 88.89% | 93.06% |
| PASQ | 89.58% | 93.06% | 89.58% | 93.06% |
| UASQ | 90.99% | 94.59% | 90.99% | 94.59% |

| v3维度 | OFF | ON |
|---|---:|---:|
| answer_correct | 64/72 | 67/72 |
| delivery_completed | 64/72 | 66/72 |
| end_to_end | 61/72 | 64/72 |

v3端到端差值 ON−OFF：+4.17个百分点，95%CSV簇探索性区间 [-6.94, +15.28] 个百分点；同源数据家族相关性仍未排除。

变更试次：通过增加0次，通过减少0次；新增模型调用0次、新增token为0。原Agent与整理器用量只计一次，不能把原分和v3成本相加。

重评分合同：`6092fc9c4b70803a2761041f6a94e14659f7170bd24b5ab10966f19e7cfd20ed`；重评分脚本：`fe01adb9623ae82a0a3d5cef2f73e15a6f4026c0aa9485141882c5e9f0b41045`。

## 环境中断批次：保留记录，不混入效果比较

以下不完整批次整批保留，不按题目成绩筛除或覆盖失败。外部主机异常与传输故障影响了运行；恢复环境后使用相同冻结代码、题目、模型和预算创建完整新批次。此表只披露运行与成本，不汇入主表成功率、配对或耗时估计。

| 批次 | 已记录／计划 | Agent已知token | Agent用量未知调用 | 未取得调用计量的试次 | 整理器已知token | 整理器未知调用 |
|---|---:|---:|---:|---:|---:|---:|
| skills-dabench-v2 | 38/144 | 1,780,314 | 15 | 0 | 21,813 | 0 |

`skills-dabench-v2`：摘要可计量的模型连接错误试次未知（38份摘要缺少该诊断字段）；最长已记录耗时23385.88秒。原因类别：host_time_jump_or_sleep、model_transport_error、docker_preflight_timeout。
其Agent token未计量试次0，整理器状态未取得的试次0；这些未知量不能当作已知零成本。原合同指纹：`4a493a176ea59eb4f9924e576425b2aea6606c6d9f6bf24b6f53d64079ca0aa7`。


## 边界与可追溯性

题目盲审风险计数：high 5题、medium 7题、low 4题、none 8题。高风险题：DA33、DA337、DA496、DA542、DA590。盲审未查看标准答案或模型结果；保留所有题目和原评分，不作豁免。

盲审后评测元数据曾变化；已核对原审计文件的suite指纹，以及两版heldout原文/约束/格式完全相同。两个原始指纹均保留，未覆盖原审计记录。

- 公开验证集的预先冻结兼容子集，不是官方隐藏测试或完整排行榜成绩。
- 同一CSV的重复运行相关；区间按CSV文件整簇重采样，是探索性区间，未排除不同CSV之间的同源相关性。
- 不同CSV文件的数量不等于相同数量的独立数据家族；CSV级开发/留出划分不保证底层数据家族完全隔离，也不能排除预训练污染。
- 开发题仅用于格式适配，没有调整Agent提示词；已知数据家族重叠仍限制效果结论的外推。
- ON比较目录、读取引导和实际正文的整体集成，不是只有正文变化的纯文本消融。
- 盲格式整理器只接收题目、格式要求与最终回答，不接收CSV、产物、标准答案或工具。
- 整理器沿用本轮固定模型，不是原论文GPT-3.5整理器的完全复现；原始分与整理后分并列。
- 题目方法、舍入和解释可能有歧义；盲审发现的问题不改题、不豁免、不提高分数。
- 失败分类是运行状态和评分字段的诊断标签，不是自动证明的根本原因。
- 未全面覆盖专业报告质量、RAG、分类/回归建模和实际业务任务；不能外推任意分析正确性。
- 技能读取次数代表实际加载，不证明遵循了指导或结果正确；未加载的ON运行仍计入全部结果。

- 数据版本：`b455d578e30fee513abd79936cbdf7a6de026cb5`。
- 评分器版本：`3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c`。
- 实现指纹：`fd967b1b19c5249a4d572bf270f1bc20f2e866022eadad05ddbae9f6dd254945`。
- 本轮合同指纹：`4a493a176ea59eb4f9924e576425b2aea6606c6d9f6bf24b6f53d64079ca0aa7`。

来源：[InfiAgent-DABench](https://infiagent.github.io/)、[官方公开数据](https://huggingface.co/datasets/infiagent/DABench)、[官方评分代码](https://github.com/InfiAgent/InfiAgent/blob/3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c/examples/DA-Agent/eval_closed_form.py)。
