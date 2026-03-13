from src.workflows.deployment_export import build_deployment_bundle


def main():
    build_deployment_bundle('configs/deployment_fine.json')
    build_deployment_bundle('configs/deployment_gross.json')
    print("model saved successfully")

if __name__ == '__main__':
    main()
