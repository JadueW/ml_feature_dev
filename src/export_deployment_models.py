from src.workflows.deployment_export import build_deployment_bundle


def main():
    fine_result = build_deployment_bundle('configs/deployment_fine.json')
    gross_result = build_deployment_bundle('configs/deployment_gross.json')
    print('Fine model:', fine_result['model_path'])
    print('Fine threshold:', fine_result['threshold'])
    print('Gross model:', gross_result['model_path'])
    print('Gross threshold:', gross_result['threshold'])


if __name__ == '__main__':
    main()
