"""多人交叠发声（crosstalk）报告核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

与 /segments 等接口不同：本接口的词序列**允许乱序与交叠**（同一时刻可能有多人
同时说话），仅要求每个词自身 end_ms >= start_ms；零时长词（end_ms == start_ms）
不参与任何统计。

算法（事件扫描线），O(n log n) 时间、O(n) 空间（n 为词数，<= 2000）：

1. 每个正时长词贡献一条左闭右开区间 [start_ms, end_ms)，生成开始/结束事件；
   同一时刻结束事件先于开始事件处理（端点相接不算同时发声）；
2. 扫描维护每个说话人的活跃区间计数，计数 >= 1 即在发声——同一说话人的
   重叠词只计一次；相邻事件点之间的正长度区间上，活跃说话人集合恒定；
3. 保留活跃说话人 >= 2 的片段，仅合并相邻且集合相同的片段（如同一说话人
   两个端点相接的词在扫描中产生的相邻片段）；
4. 每段活跃集合的每个无序说话人对都累加该段时长——三人同时发声的时长
   分别计入三对。
"""

from itertools import combinations

from .errors import DomainError

MAX_SPEAKERS = 20  # 不同说话人上限

_END, _START = 0, 1  # 事件类型序：同一时刻结束事件先于开始事件处理


def _fail(code, message, **extra):
    raise DomainError(code, message, **extra)


def crosstalk_report(words):
    """统计多人同时发声的片段与各说话人对的总重叠毫秒数。

    参数 words 中每个对象需具备属性：start_ms / end_ms / speaker（text 本接口
    不使用）。词序列允许乱序与交叠；end_ms 早于 start_ms 时抛出
    DomainError(code="invalid_time_range")；不同说话人超过 20 个时抛出
    DomainError(code="too_many_speakers")。

    返回 (segments, pairs)：segments 按时间升序，每项为 dict：
    index / start_ms / end_ms / speakers（按标识升序，至少 2 人）；
    pairs 按说话人标识排序，每项为 dict：speakers（2 个标识，升序）/ overlap_ms。
    """
    if not words:
        _fail("empty_words", "words 不能为空")

    events = []
    speakers = set()
    for idx, w in enumerate(words):
        if w.end_ms < w.start_ms:
            _fail(
                "invalid_time_range",
                f"第 {idx} 个词的 end_ms({w.end_ms}) 早于 start_ms({w.start_ms})",
                word_index=idx,
            )
        speakers.add(w.speaker)
        if w.end_ms == w.start_ms:
            continue  # 零时长词不参与统计
        events.append((w.start_ms, _START, w.speaker))
        events.append((w.end_ms, _END, w.speaker))

    if len(speakers) > MAX_SPEAKERS:
        _fail(
            "too_many_speakers",
            f"不同说话人共 {len(speakers)} 个，超过上限 {MAX_SPEAKERS}",
            speaker_count=len(speakers),
        )

    events.sort(key=lambda e: (e[0], e[1]))

    # ---- 扫描：活跃说话人集合恒定的原子片段（仅保留 >= 2 人的） ----
    pieces = []  # (start_ms, end_ms, 活跃说话人升序元组)
    active = {}  # 说话人 -> 当前覆盖该说话人的区间数（>= 1 即在发声）
    prev_t = None
    for t, kind, speaker in events:
        if prev_t is not None and t > prev_t and len(active) >= 2:
            pieces.append((prev_t, t, tuple(sorted(active))))
        if kind == _START:
            active[speaker] = active.get(speaker, 0) + 1
        else:
            count = active[speaker] - 1
            if count:
                active[speaker] = count
            else:
                del active[speaker]
        prev_t = t

    # ---- 仅合并相邻且集合相同的片段 ----
    segments = []
    for start, end, group in pieces:
        if segments and segments[-1][2] == group and segments[-1][1] == start:
            segments[-1][1] = end
        else:
            segments.append([start, end, group])

    # ---- 说话人对总重叠：三人同时发声的时长分别计入三对 ----
    pair_overlap = {}
    for start, end, group in segments:
        for pair in combinations(group, 2):
            pair_overlap[pair] = pair_overlap.get(pair, 0) + (end - start)

    segment_dicts = [
        {"index": idx, "start_ms": start, "end_ms": end, "speakers": list(group)}
        for idx, (start, end, group) in enumerate(segments)
    ]
    pair_dicts = [
        {"speakers": list(pair), "overlap_ms": overlap}
        for pair, overlap in sorted(pair_overlap.items())
    ]
    return segment_dicts, pair_dicts
