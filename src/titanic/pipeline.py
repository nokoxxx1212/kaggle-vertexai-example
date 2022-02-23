from kfp.v2 import dsl, compiler, components

# CONSTANTS
gcp_gcs_pipeline_root = ${gcp_gcs_pipeline_root}
raw_data_path = '/input/titanic'
transformed_data_path = '/output/titanic'
trained_model_path = '/output/titanic'
predicted_data_path = '/output/titanic'

@dsl.pipeline(name='pipeline', pipeline_root=gcp_gcs_pipeline_root)
def pipeline():
    # transform
    # コンポーネントの読み込み
    transform_op = components.load_component_from_file(
        'components/transform/component.yaml')
    # コンポーネントの実行
    transform_task = transform_op(raw_data_path, transformed_data_path)

    # trainer
    # コンポーネントの読み込み
    trainer_op = components.load_component_from_file(
        'components/trainer/component.yaml')
    # コンポーネントの実行
    trainer_task = trainer_op(transformed_data_path, trained_model_path)

    # predictor
    # コンポーネントの読み込み
    predictor_op = components.load_component_from_file(
        'components/predictor/component.yaml')
    # コンポーネントの実行
    predictor_task = predictor_op(transformed_data_path, trained_model_path, predicted_data_path)


### パイプライン関数をコンパイルする
compiler.Compiler().compile(pipeline_func=pipeline,
                            package_path='pipeline.json')