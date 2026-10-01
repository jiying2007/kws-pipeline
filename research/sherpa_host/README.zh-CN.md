# 独立 Linux host KWS 研究入口

这是已验证 B 候选的源码配方：KWS-root 整 TU 子集、14 个 C API、固定有界
Viterbi 策略与文件 CLI。它不接入默认 C11 产品、模型注册表或旧 sherpa_pcm
适配器；不能据此宣称声学质量、近音拒绝、延迟或 SSC305 资格改善。

仓库只新增源码文本、完整 decoder 补丁、精确依赖/选源清单和离线测试。
上游源、ORT、模型、音频均为外部输入。构建步骤见 [BUILDING.md](BUILDING.md)，
实测边界见 [RESULTS.md](RESULTS.md)，版权与来源见 [PROVENANCE.md](PROVENANCE.md)。

## 默认离线测试

在仓库根使用 Python 3 与 GCC C++17：

```sh
python3 research/sherpa_host/tests/run_tests.py --sanitize
```

测试真实 CLI 的解析、JSON、参数、session 与对象释放 helper，使用 fake C API；
另测 launcher 搬移/参数/完整性、wrapper 的19个原片段与14导出、源/许可/编译
合同及 materializer 负例。小波形由固定数字临时生成。默认 CI 不编译完整
wrapper/decoder、不加载 ORT/模型，也不声称已执行 Viterbi、Hypothesis 或图回归。
这类测试需明确提供锁定 source/ORT。根六个 CI context 保持不变。

## 使用已审运行目录

```sh
./run-kws --wav -- "relative directory/audio.wav"
./run-kws --pcm-s16le-16000-mono -- "-audio.pcm"
```

一次读取一个不超过1 GiB的普通文件，默认 WAV。只接受小端 RIFF/WAVE、PCM1、
单声道16 kHz/16 bit；fmt/data各一个，fmt长16或18且扩展长度为零，RIFF长度
严格等于文件大小，chunk和奇数padding完整。拒绝空音频、奇数PCM字节、重复/
截断chunk、尾随字节、压缩/浮点/RF64/RIFX、stdin/FIFO/设备。裸PCM需显式选项。
不重采样、混声道或补静音。末块不足320样本照实送入，InputFinished后排空结果。

相对输入以调用者当前目录为准；路径空格需加引号，以横线开头需使用 `--`。
套件可整体搬移。处理期间不要修改输入，也不要把输出重定向到输入；长度检查
无法发现等长并发改写。固定CPU、单线程、80维、320样本块、max_active_paths=4、
trailing_blanks=1、score=1、threshold=0.25及两词研究配置，不提供调参覆盖。

stdout为NDJSON：event含keyword/tokens/timestamps/start_time/available_samples/eof，
正常结束另发complete（输入字节、样本、feed和event数量）。API时间值原样保留；
available_samples表示已送样本，不是声学延迟。必须检查退出码与complete。
非法UTF-8、非有限数、超限结果及输出失败返回非零，可能已有部分输出；错误到stderr。

launcher先核验全清单，再以最小环境执行；用于避免误混依赖，不是签名或恶意
loader安全沙箱。修改模型、参数或二进制必须重新建立身份与验证。

## 平台与资格

当前实测是现代Linux x86-64；版本需求包括GLIBC_2.38、GLIBCXX_3.4.32、
CXXABI_1.3.15，仍须满足完整依赖/符号集合。B/ORT最低ISA未知，CLI的baseline
标记不能外推整个运行包。需要GNU coreutils与/proc/self/exe；不是通用Linux或
Windows二进制。SSC305 A32需要真实SDK/sysroot/loader和独立实机验收。

子集库SONAME为libsherpa-onnx-kws-c-api.so，仅实现导出表列出的14个函数。
复用的完整上游公开头包含额外声明，不得调用未实现API或与全量库混用opaque
handles。仍保留模型工厂、sentencepiece/FasterDecoder/FST/Eigen等传递依赖。
同Viterbi工程一致性不等于原官方算法一致性；历史官方严格parity失败保持不变。
