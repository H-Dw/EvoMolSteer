# 固定预实验scale的梯度原子范围实验与08:00归档

2026-11-01，Asia/Shanghai。用户授权在baseline配置上比较condition、condition+1跳、condition+2跳，共三种范围、两个教师；不使用Guidance Agent，不迭代。另要求上海时间08:00激活结果打包任务，将不含模型权重的tar.gz放入/root/private_data。

## 科学定义

仅限制条件绝对坐标loss产生的额外坐标VJP/引导增量。FLOWR原有的速度、漂移、噪声、分类采样、自条件及全体原子的原生更新保留。条件loss仍只包含原九个PLIP直接互作重原子，sum(error²)/(2K)，不把邻居加入loss，不增加亲和力或相对距离项。

每步先按baseline动态Hungarian得到该步loss的condition生成槽位；0跳只允许这些槽位接收额外梯度。1/2跳使用同一步预测终态的原生五类bonds.argmax，类别0为无键，取有向预测的无向并集，从该步condition槽位作累计一/二跳扩展；排除自身环和padding。先完整计算原VJP，再对范围外额外梯度严格置0，不截断网络反向传播。终态仍动态匹配。原始VJP、实际掩码、范围外增量最大值逐步保存，范围外guided_proposal须逐项等于native_proposal。

保持baseline四窗口10–25/26–50/51–75/76–99，无联合前缀，10–99化学状态inpainting，原九原子诱导键注入区域不扩大。raw999、完整100区间0–0.999、第99步更新不进入返回预测、完整预测终态VJP、蛋白单独确定测量变换不变。没有3Å限幅、向量缩放、高RMSD梯度掩码或终态坐标覆盖。

## 固定scale来源

只读原baseline任务的calibration/domains.json及initial_source完整人口回执：6pyy参考为70，3w54参考为90，四窗口分别[70,70,70,70]和[90,90,90,90]。不使用baseline后续Agent范围取证选出的90等参数，也不使用prefix/five_windows的Agent最优参数。部署记录原domains文件SHA、原初始源COMPLETE及全部产物SHA。新掩码策略未重新标定，明确是固定baseline预标定参数的干预实验，不能宣称已为新策略重新确认安全域。

六任务依次为condition_6pyy、condition_3w54、hop1_6pyy、hop1_3w54、hop2_6pyy、hop2_3w54。每任务新根、prepare→新提交自身完整acceptance→run-fixed-scope一次50候选→CPU固定10/50诊断。总科学预算300，标定新增0、Agent/LLM调用0、迭代0。相同五种子各10槽位和原冻结初态用于配对复现，不是独立验证。失败计费、不补样、不降档、不自动重启；单任务失败保留后继续其他任务。

UG04串行调度保持：逻辑组50、网络/VJP固定10行默认stream顺序执行、CPU算子8线程；无GPU线程池、额外stream或GPU子进程池。独立CPU质量worker隐藏GPU、数值库1线程。每组抽查五种子slot0/1共10，其余40不追加慢查；构建失败占分母，PLIP/PoseB内部几何、蛋白冲突、能量分报，不外推全人口。

## 验收与实现

新增可选guidance_atom_scope接口，默认all路径保持原张量，无掩码乘法。显式condition_hops仅允许整数0、1、2，拒绝bool、未知字段、重复/越界/padding条件槽位。掩码、图定义及新模块哈希进入兼容绑定与检查点，禁止把新旧采样回执互换。专用fixed模式要求rounds=0、固定四窗口、baseline来源身份及SHA、准确原预标定值，并禁止run-version调用Agent。

CPU验收覆盖图半径、候选独立、方向并集、padding、动态槽位、完整耦合VJP后置掩码、选中梯度不变、范围外额外引导为0、原生运动不变及无额外RNG。新提交在远端独立检出，运行相关CPU回归；每任务自己的真实模型完整acceptance不能由旧回执或CPU测试替代。nonfinite_isolation仍为预期夹具。

## 08:00 tar.gz归档

目标启动时间为2026-11-01 08:00，Asia/Shanghai；这是一次归档任务，完成后删除本专项定时触发。范围为本对话当前可见的实验结果根以及本次新增六任务根，实际根清单写ARCHIVE_PLAN.json；不触碰其他项目或运行中的任务。

打包结果、HDF5轨迹、终态/检查点张量、结构、账本、claims、LLM回执、日志与审计。排除模型权重、配置指定的模型checkpoint、launcher.log、core及已有备份tar.gz，防止递归重复打包；实验checkpoint_010.pt.gz、checkpoint_100.pt.gz与terminal.pt.gz仍属于结果，必须保留。模型和外部输入的符号链接只记录引用，不解引用复制权重。

先在空间充足的/workspace打包。逐文件核对打包前后size/mtime/内容SHA，包含失败和部分保存产物；活动HDF5不打包，检测到写入时停止并保留证据。归档附逐成员SHA清单，完整校验后按实际可用空间复制/移动tar.gz到/root/private_data，目的副本SHA一致后才删除临时tar.gz；原始实验文件不删除。

03:32实际核查：/root/private_data仅余2,066,743,296字节；最近两轮已有压缩包合计4,344,429,977字节，原始结果约5.81GB，尚不含更早根和本次新实验。已向用户询问扩容、其他持久路径或缩小范围，等待回复；未经回复不缩小“所有结果”、不删除旧文件腾空间。08:00可先在/workspace完成压缩和校验，若实际tar.gz仍大于目标剩余空间，保留已验证tar.gz于/workspace并报告阻塞，不能冒称已移到持久盘。

中文UTF-8，所有时间标Asia/Shanghai。仅在codex/uniguide-flowr-agent修改并提交推送，新固定目录运行；既有结果、失败、FINALIZED、已完成四项与两个重测只读保留。原uniguide-flowr继续PAUSED/35分钟，不恢复旧巡检，不向其他聊天发消息。凭据不输出或写入仓库/自动化，所有launcher.log及core不下载。08:00任务实际读取新六项进度和结果，收集审计；仅在路径/进程归属、质量队列空闲确认后停止本次CPU worker。
