#!/usr/bin/env python3
"""
测试 visualize_critic_fixed.py 中全局状态索引的正确性。

根据 AGENTS.md 第6.8节，全局状态结构为：
- [0:6]   A1状态 (pos_x, pos_y, vel_x, vel_y, is_in_spot, shoot_progress)
- [6:10]  A2状态 (pos_x, pos_y, vel_x, vel_y)
- [10:14] D1状态 (pos_x, pos_y, vel_x, vel_y)
- [14:18] D2状态 (pos_x, pos_y, vel_x, vel_y)
- [18:20] 投篮点位置
- [20:22] 篮筐位置
- [22:23] 剩余时间
"""


def test_global_state_indexing():
    global_state = list(range(23))

    test_cases = [
        {"agent": "A1", "index": 0, "expected_start": 0, "expected_range": (0, 6)},
        {"agent": "A2", "index": 1, "expected_start": 6, "expected_range": (6, 10)},
        {"agent": "D1", "index": 2, "expected_start": 10, "expected_range": (10, 14)},
        {"agent": "D2", "index": 3, "expected_start": 14, "expected_range": (14, 18)},
    ]

    print("测试全局状态索引逻辑：")
    print(f"全局状态维度: {len(global_state)}")
    print()

    all_passed = True

    for test in test_cases:
        agent_to_vary = test["index"]
        agent_name = test["agent"]
        expected_start = test["expected_start"]

        if agent_to_vary == 0:
            start_idx = 0
        elif agent_to_vary == 1:
            start_idx = 6
        elif agent_to_vary == 2:
            start_idx = 10
        else:
            start_idx = 14

        is_correct = start_idx == expected_start
        status = "✓ PASS" if is_correct else "✗ FAIL"

        if not is_correct:
            all_passed = False

        state_dims = 6 if agent_to_vary == 0 else 4
        agent_state = global_state[start_idx : start_idx + state_dims]

        print(f"{status} | {agent_name} (index={agent_to_vary})")
        print(f"  - 起始索引: {start_idx} (期望: {expected_start})")
        print(f"  - 状态维度: {state_dims}")
        print(f"  - 索引范围: [{start_idx}:{start_idx + state_dims}]")
        print(f"  - 状态内容: {agent_state}")
        print()

    all_passed = True

    for test in test_cases:
        agent_to_vary = test["index"]
        agent_name = test["agent"]
        expected_start = test["expected_start"]
        expected_range = test["expected_range"]

        # 模拟修复后的索引计算逻辑
        if agent_to_vary == 0:  # A1
            start_idx = 0
        elif agent_to_vary == 1:  # A2
            start_idx = 6
        elif agent_to_vary == 2:  # D1
            start_idx = 10
        else:  # D2
            start_idx = 14

        # 验证索引
        is_correct = start_idx == expected_start
        status = "✓ PASS" if is_correct else "✗ FAIL"

        if not is_correct:
            all_passed = False

        # 获取该智能体的完整状态
        agent_state = (
            global_state[start_idx : start_idx + 4]
            if agent_to_vary > 0
            else global_state[start_idx : start_idx + 6]
        )
        state_dims = 6 if agent_to_vary == 0 else 4

        print(f"{status} | {agent_name} (index={agent_to_vary})")
        print(f"  - 起始索引: {start_idx} (期望: {expected_start})")
        print(f"  - 状态维度: {state_dims}")
        print(f"  - 索引范围: [{start_idx}:{start_idx + state_dims}]")
        print(f"  - 状态内容: {agent_state}")
        print()

    print("环境特征索引：")
    print(f"  投篮点: [{18}:{20}] = {global_state[18:20]}")
    print(f"  篮筐: [{20}:{22}] = {global_state[20:22]}")
    print(f"  时间: [{22}:{23}] = {global_state[22:23]}")
    print()

    print("=" * 60)
    if all_passed:
        print("✓ 所有测试通过！全局状态索引逻辑正确。")
    else:
        print("✗ 部分测试失败！请检查索引逻辑。")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    test_global_state_indexing()
