"""多人交叠发声报告测试：核心算法、暴力对拍与 HTTP 接口层。"""

import random
import unittest
from itertools import combinations
from types import SimpleNamespace

from app.crosstalk_report import crosstalk_report
from app.errors import DomainError


def W(start, end, speaker, text="词"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def segments_and_pairs(words):
    return crosstalk_report(words)


class CrosstalkReportCoreTests(unittest.TestCase):
    def test_basic_two_speakers(self):
        # A [0,2000)、B [1500,3500)：交叠 [1500,2000) 共 500ms
        segments, pairs = segments_and_pairs([W(0, 2000, "A"), W(1500, 3500, "B")])
        self.assertEqual(
            segments,
            [{"index": 0, "start_ms": 1500, "end_ms": 2000, "speakers": ["A", "B"]}],
        )
        self.assertEqual(pairs, [{"speakers": ["A", "B"], "overlap_ms": 500}])

    def test_words_may_be_out_of_order(self):
        ordered = [W(0, 2000, "A"), W(1500, 3500, "B")]
        shuffled = [W(1500, 3500, "B"), W(0, 2000, "A")]
        self.assertEqual(segments_and_pairs(ordered), segments_and_pairs(shuffled))

    def test_words_may_overlap_across_and_within_speakers(self):
        # A 的两个词互相重叠，覆盖 [0,1500) 只计一次；B [800,1200)
        words = [W(0, 1000, "A"), W(500, 1500, "A"), W(800, 1200, "B")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(
            segments,
            [{"index": 0, "start_ms": 800, "end_ms": 1200, "speakers": ["A", "B"]}],
        )
        self.assertEqual(pairs, [{"speakers": ["A", "B"], "overlap_ms": 400}])

    def test_same_speaker_identical_words_counted_once(self):
        # 同一说话人两个完全相同的词，B 只与其中“一份覆盖”相交
        words = [W(0, 1000, "A"), W(0, 1000, "A"), W(300, 700, "B")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(
            segments,
            [{"index": 0, "start_ms": 300, "end_ms": 700, "speakers": ["A", "B"]}],
        )
        self.assertEqual(pairs, [{"speakers": ["A", "B"], "overlap_ms": 400}])

    def test_touching_endpoints_not_overlap(self):
        # A 恰在 B 起点结束（左闭右开），以及 B 恰在 A 起点结束
        words = [W(0, 1000, "A"), W(1000, 2000, "B")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(segments, [])
        self.assertEqual(pairs, [])

        words = [W(1000, 2000, "A"), W(0, 1000, "B")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(segments, [])
        self.assertEqual(pairs, [])

    def test_adjacent_same_set_pieces_merged(self):
        # A 有两个端点相接的词 [0,1000)、[1000,2000)，B [500,1500)：
        # 扫描会在 1000 处切出两个相邻且集合同为 {A,B} 的片段，须合并
        words = [W(0, 1000, "A"), W(1000, 2000, "A"), W(500, 1500, "B")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(
            segments,
            [{"index": 0, "start_ms": 500, "end_ms": 1500, "speakers": ["A", "B"]}],
        )
        self.assertEqual(pairs, [{"speakers": ["A", "B"], "overlap_ms": 1000}])

    def test_three_speakers_counts_three_pairs(self):
        # A [0,1000)、B [200,900)、C [400,1200)
        # 三人段 [400,900) 500ms 分别计入 AB/AC/BC；另 AB 200ms、AC 100ms
        words = [W(0, 1000, "A"), W(200, 900, "B"), W(400, 1200, "C")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(
            segments,
            [
                {"index": 0, "start_ms": 200, "end_ms": 400, "speakers": ["A", "B"]},
                {"index": 1, "start_ms": 400, "end_ms": 900, "speakers": ["A", "B", "C"]},
                {"index": 2, "start_ms": 900, "end_ms": 1000, "speakers": ["A", "C"]},
            ],
        )
        self.assertEqual(
            pairs,
            [
                {"speakers": ["A", "B"], "overlap_ms": 700},
                {"speakers": ["A", "C"], "overlap_ms": 600},
                {"speakers": ["B", "C"], "overlap_ms": 500},
            ],
        )

    def test_disjoint_crosstalk_regions_accumulate_pair_total(self):
        # AB 两段交叠 [0,100) 与 [500,600)，中间隔着单人段，不能跨空档合并
        words = [
            W(0, 100, "A"), W(50, 100, "B"),
            W(200, 600, "A"), W(500, 600, "B"),
        ]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(
            segments,
            [
                {"index": 0, "start_ms": 50, "end_ms": 100, "speakers": ["A", "B"]},
                {"index": 1, "start_ms": 500, "end_ms": 600, "speakers": ["A", "B"]},
            ],
        )
        self.assertEqual(pairs, [{"speakers": ["A", "B"], "overlap_ms": 150}])

    def test_speakers_and_pairs_sorted_by_identifier(self):
        words = [
            W(0, 100, "B"),
            W(0, 100, "C"),
            W(0, 100, "A"),
        ]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(segments[0]["speakers"], ["A", "B", "C"])
        self.assertEqual(
            pairs,
            [
                {"speakers": ["A", "B"], "overlap_ms": 100},
                {"speakers": ["A", "C"], "overlap_ms": 100},
                {"speakers": ["B", "C"], "overlap_ms": 100},
            ],
        )

    def test_zero_duration_words_ignored(self):
        # 全部为零时长：无统计结果
        segments, pairs = segments_and_pairs([W(0, 0, "A"), W(0, 0, "B")])
        self.assertEqual(segments, [])
        self.assertEqual(pairs, [])

        # 零时长词落在他人区间内，不凭空增加说话人
        words = [W(0, 100, "A"), W(0, 100, "B"), W(50, 50, "C")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(
            segments,
            [{"index": 0, "start_ms": 0, "end_ms": 100, "speakers": ["A", "B"]}],
        )
        self.assertEqual(pairs, [{"speakers": ["A", "B"], "overlap_ms": 100}])

    def test_single_speaker_has_no_crosstalk(self):
        words = [W(0, 100, "A"), W(50, 150, "A"), W(150, 300, "A")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(segments, [])
        self.assertEqual(pairs, [])

    def test_no_overlap(self):
        words = [W(0, 100, "A"), W(200, 300, "B"), W(400, 500, "A")]
        segments, pairs = segments_and_pairs(words)
        self.assertEqual(segments, [])
        self.assertEqual(pairs, [])

    def test_end_before_start_raises_domain_error(self):
        with self.assertRaises(DomainError) as ctx:
            crosstalk_report([W(0, 100, "A"), W(300, 200, "B")])
        self.assertEqual(ctx.exception.code, "invalid_time_range")
        self.assertEqual(ctx.exception.extra["word_index"], 1)

    def test_too_many_speakers(self):
        words = [W(0, 1, f"S{i}") for i in range(21)]
        with self.assertRaises(DomainError) as ctx:
            crosstalk_report(words)
        self.assertEqual(ctx.exception.code, "too_many_speakers")
        self.assertEqual(ctx.exception.extra["speaker_count"], 21)

        # 恰好 20 个不报错（20 人同时发声：1 段、C(20,2)=190 对）
        words = [W(0, 1, f"S{i:02d}") for i in range(20)]
        segments, pairs = crosstalk_report(words)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["speakers"], [f"S{i:02d}" for i in range(20)])
        self.assertEqual(len(pairs), 190)

    def test_speaker_limit_counts_zero_duration_speakers_too(self):
        # 零时长词不参与统计，但其说话人仍计入不同说话人数
        words = [W(0, 1, "A"), W(0, 0, "B")]
        for i in range(19):
            words.append(W(0, 0, f"S{i}"))
        with self.assertRaises(DomainError) as ctx:
            crosstalk_report(words)
        self.assertEqual(ctx.exception.code, "too_many_speakers")
        self.assertEqual(ctx.exception.extra["speaker_count"], 21)

    def test_empty_words_raises(self):
        with self.assertRaises(DomainError) as ctx:
            crosstalk_report([])
        self.assertEqual(ctx.exception.code, "empty_words")


class CrosstalkReportDifferentialTests(unittest.TestCase):
    """逐毫秒枚举活跃集合的暴力对拍（所有时间均为整数，逐毫秒采样对左闭右开
    整数区间是精确的）。"""

    def brute_force(self, words):
        horizon = max((w.end_ms for w in words if w.end_ms > w.start_ms), default=0)
        raw = []
        for t in range(horizon):
            active = {
                w.speaker
                for w in words
                if w.start_ms <= t < w.end_ms and w.end_ms > w.start_ms
            }
            raw.append(active)

        expected_segments = []
        for t, active in enumerate(raw):
            group = tuple(sorted(active)) if len(active) >= 2 else None
            if group is None:
                continue
            if expected_segments and expected_segments[-1][2] == group \
                    and expected_segments[-1][1] == t:
                expected_segments[-1][1] = t + 1
            else:
                expected_segments.append([t, t + 1, group])

        expected_pairs = {}
        for active in raw:
            if len(active) >= 2:
                for pair in combinations(sorted(active), 2):
                    expected_pairs[pair] = expected_pairs.get(pair, 0) + 1

        return expected_segments, expected_pairs

    def test_random_cases(self):
        rng = random.Random(20260925)
        for trial in range(500):
            n = rng.randint(1, 10)
            pool = ["A", "B", "C", "D"][: rng.randint(1, 4)]
            words = []
            for _ in range(n):
                start = rng.randint(0, 24)
                end = start + rng.randint(0, 6)  # 允许零时长
                words.append(W(start, end, rng.choice(pool), text="词"))
            rng.shuffle(words)  # 强制乱序输入

            segments, pairs = crosstalk_report(list(words))
            exp_segments, exp_pairs = self.brute_force(words)

            self.assertEqual(
                segments,
                [
                    {"index": i, "start_ms": s, "end_ms": e, "speakers": list(g)}
                    for i, (s, e, g) in enumerate(exp_segments)
                ],
                msg=f"trial {trial}: {[(w.start_ms, w.end_ms, w.speaker) for w in words]}",
            )
            self.assertEqual(
                pairs,
                [
                    {"speakers": list(p), "overlap_ms": d}
                    for p, d in sorted(exp_pairs.items())
                ],
                msg=f"trial {trial}: {[(w.start_ms, w.end_ms, w.speaker) for w in words]}",
            )


try:
    from fastapi.testclient import TestClient
    from app.main import app
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过接口层测试")
class CrosstalkReportApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_basic_report(self):
        resp = self.client.post(
            "/crosstalk-report",
            json={
                "words": [
                    {"start_ms": 1500, "end_ms": 3500, "text": "插话", "speaker": "B"},
                    {"start_ms": 2500, "end_ms": 4000, "text": "继续讲", "speaker": "A"},
                    {"start_ms": 0, "end_ms": 2000, "text": "开场白", "speaker": "A"},
                    {"start_ms": 3000, "end_ms": 3600, "text": "附和", "speaker": "C"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {
                "segments": [
                    {"index": 0, "start_ms": 1500, "end_ms": 2000, "speakers": ["A", "B"]},
                    {"index": 1, "start_ms": 2500, "end_ms": 3000, "speakers": ["A", "B"]},
                    {"index": 2, "start_ms": 3000, "end_ms": 3500, "speakers": ["A", "B", "C"]},
                    {"index": 3, "start_ms": 3500, "end_ms": 3600, "speakers": ["A", "C"]},
                ],
                "pairs": [
                    {"speakers": ["A", "B"], "overlap_ms": 1500},
                    {"speakers": ["A", "C"], "overlap_ms": 600},
                    {"speakers": ["B", "C"], "overlap_ms": 500},
                ],
            },
        )

    def test_empty_result(self):
        resp = self.client.post(
            "/crosstalk-report",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 100, "text": "甲", "speaker": "A"},
                    {"start_ms": 100, "end_ms": 200, "text": "乙", "speaker": "B"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"segments": [], "pairs": []})

    def test_zero_duration_words_accepted(self):
        resp = self.client.post(
            "/crosstalk-report",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 0, "text": "甲", "speaker": "A"},
                    {"start_ms": 0, "end_ms": 0, "text": "乙", "speaker": "B"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"segments": [], "pairs": []})

    def test_end_before_start_returns_422(self):
        resp = self.client.post(
            "/crosstalk-report",
            json={
                "words": [
                    {"start_ms": 0, "end_ms": 100, "text": "甲", "speaker": "A"},
                    {"start_ms": 300, "end_ms": 200, "text": "乙", "speaker": "B"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "invalid_time_range")
        self.assertEqual(err["word_index"], 1)

    def test_overlap_and_disorder_allowed(self):
        # 同样的输入在 /segments 会因重叠 422，但本接口允许
        resp = self.client.post(
            "/crosstalk-report",
            json={
                "words": [
                    {"start_ms": 1500, "end_ms": 3000, "text": "乙", "speaker": "B"},
                    {"start_ms": 0, "end_ms": 2000, "text": "甲", "speaker": "A"},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["segments"]), 1)
        self.assertEqual(data["segments"][0]["speakers"], ["A", "B"])

    def test_too_many_words_returns_422(self):
        words = [
            {"start_ms": 0, "end_ms": 1, "text": "词", "speaker": "A"}
            for _ in range(2001)
        ]
        resp = self.client.post("/crosstalk-report", json={"words": words})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")

    def test_too_many_speakers_returns_422(self):
        words = [
            {"start_ms": 0, "end_ms": 1, "text": "词", "speaker": f"S{i}"}
            for i in range(21)
        ]
        resp = self.client.post("/crosstalk-report", json={"words": words})
        self.assertEqual(resp.status_code, 422)
        err = resp.json()["error"]
        self.assertEqual(err["code"], "too_many_speakers")
        self.assertEqual(err["speaker_count"], 21)

    def test_empty_words_returns_422(self):
        resp = self.client.post("/crosstalk-report", json={"words": []})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
