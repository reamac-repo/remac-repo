def merge_plans_with_last(history_plans):
    final_plan = []
    for plan in history_plans:
        if plan:  # check whether plan is empty
            current_action = plan[0]
            if not final_plan or current_action != final_plan[-1]:
                final_plan.append(current_action)
    
    # handle last plan
    if history_plans:
        last_plan = history_plans[-1]
        if len(last_plan) > 0:
            remaining_actions = last_plan[1:]
            for action in remaining_actions:
                if not final_plan or action != final_plan[-1]:
                    final_plan.append(action)
    
    return final_plan


def merge_plans_without_last(history_plans):
    final_plan = []
    for plan in history_plans:
        if plan:  # check whether plan is empty
            current_action = plan[0]
            if not final_plan or current_action != final_plan[-1]:
                final_plan.append(current_action)
    
    return final_plan


if __name__ == '__main__':
    history_plan = [
        ["str1", "str2", "str3", "str4", "str5", "str6"],
        ["str2", "str3", "str4", "str5", "str6"],
        ["str2", "str3", "str4", "str5", "str6"],
        ["str3", "str4", "str5", "str6"],
        ["str4", "str1", "str2", "str5", "str6"],
        ["str1", "str2", "str5", "str6"],
        ["str2", "str5", "str6"],
        ["str5", "str6"]
    ]
    merged = merge_plans_without_last(history_plan)
    print(merged)