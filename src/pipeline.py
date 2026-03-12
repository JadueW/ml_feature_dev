from src.workflows.offline_runner import run_offline_experiments


def main():
    results = run_offline_experiments('configs/baseline_v2.json')
    print('Experiment:', results['experiment_name'])
    if 'leave_one_day_out' in results:
        print('LODO mean AUC:', results['leave_one_day_out'].get('test_auc_mean'))
        print('LODO mean Balanced Accuracy:', results['leave_one_day_out'].get('test_balanced_accuracy_mean'))
    if 'chronological' in results:
        print('Chronological mean AUC:', results['chronological'].get('test_auc_mean'))
        print('Chronological mean Balanced Accuracy:', results['chronological'].get('test_balanced_accuracy_mean'))


if __name__ == '__main__':
    main()
