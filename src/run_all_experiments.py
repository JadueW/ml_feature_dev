from src.workflows.offline_runner import run_offline_experiments


CONFIGS = [
    'configs/baseline_v2_fineMovement.json',
    'configs/baseline_v2_grossMovement.json'
]


def main():
    all_results = []
    for config_path in CONFIGS:
        result = run_offline_experiments(config_path)
        all_results.append(result)
        print('Finished:', result['experiment_name'])
        if 'leave_one_day_out' in result:
            print('  LODO mean AUC:', result['leave_one_day_out'].get('test_auc_mean'))
            print('  LODO mean Balanced Accuracy:', result['leave_one_day_out'].get('test_balanced_accuracy_mean'))
        if 'chronological' in result:
            print('  Chronological mean AUC:', result['chronological'].get('test_auc_mean'))
            print('  Chronological mean Balanced Accuracy:', result['chronological'].get('test_balanced_accuracy_mean'))
    return all_results


if __name__ == '__main__':
    main()
