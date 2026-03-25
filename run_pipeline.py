import gc
    import os

    import numpy as np
    import pandas as pd

    from src.config import N_CLUSTERS, OUTDIR, SAMPLE_SUBMISSION_PATH, SUBMISSION_PATH, TEST_PATH, TRAIN_PATH
    from src.features import (
        add_location_features,
        add_solar_geometry_approx,
        add_time_and_interactions,
        encode_identifiers,
        select_base_features,
    )
    from src.modeling import build_train_valid_test, evaluate_model, train_lightgbm
    from src.postprocessing import apply_postprocessing
    from src.preprocessing import (
        apply_midpoint_fill,
        ensure_output_dir,
        normalize_missing,
        optimize_dtypes,
        print_header,
        print_memory_usage,
    )
    from src.visualization import (
        plot_feature_importance,
        plot_location_map,
        plot_prediction_scatter,
        save_feature_importance,
    )


    def main() -> None:
        ensure_output_dir()
        print_header()

        print('
📥 Loading CSVs...')
        print_memory_usage('start')

        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        print(f'✅ train: {len(train):,}행 x {len(train.columns)}열')
        print(f'✅ test : {len(test):,}행 x {len(test.columns)}열')
        print_memory_usage('loaded')

        if 'energy' in train.columns:
            print("⚠️ Removing 'energy' (not in test)")
            train = train.drop(columns=['energy'])

        train = optimize_dtypes(train)
        test = optimize_dtypes(test)
        train = normalize_missing(train)
        test = normalize_missing(test)

        train['time'] = pd.to_datetime(train['time'])
        test['time'] = pd.to_datetime(test['time'])

        train, test = apply_midpoint_fill(train, test)

        train = add_time_and_interactions(train)
        test = add_time_and_interactions(test)

        train, test, pv_coords_info, cluster_centers = add_location_features(train, test, n_clusters=N_CLUSTERS)

        train = add_solar_geometry_approx(train)
        test = add_solar_geometry_approx(test)

        train, test = encode_identifiers(train, test)

        features_base = select_base_features(train, test)
        print(f'
✅ Base features: {len(features_base)}개')

        x_train, x_val, y_train, y_val, test_feat, pv_tr, pv_va = build_train_valid_test(train, test, features_base)
        features = list(x_train.columns)
        print(f'✅ Total features (with pv_te): {len(features)}')

        model = train_lightgbm(x_train, y_train, x_val, y_val)
        metrics = evaluate_model(model, x_train, y_train, x_val, y_val)

        plot_prediction_scatter(model, x_val, y_val, metrics['best_iteration'], OUTDIR)

        importance_df = save_feature_importance(model, features, OUTDIR)
        plot_feature_importance(importance_df, features, OUTDIR)
        plot_location_map(pv_coords_info, cluster_centers, pv_tr, pv_va, OUTDIR)

        print('
[예측/후처리 중...]')
        raw_pred = np.maximum(0, model.predict(test_feat, num_iteration=metrics['best_iteration']))
        post_pred = apply_postprocessing(test, raw_pred)

        print(
            f"[Check] zeros(≤1e-06): {np.mean(post_pred <= 1e-6) * 100:.2f}%, "
            f"min={post_pred.min():.2f}, max={post_pred.max():.2f}"
        )

        submission = submission.copy()
        submission['nins'] = post_pred
        submission.to_csv(SUBMISSION_PATH, index=False, float_format='%.2f')
        print(f'
✅ Saved: {SUBMISSION_PATH}')
        print(f'   Size: {os.path.getsize(SUBMISSION_PATH) / 1024 / 1024:.2f} MB')

        summary = {
            'tweedie_power': 1.1,
            'n_estimators': 25000,
            'best_iter': int(metrics['best_iteration']),
            'learning_rate': 0.02,
            'early_stopping': 500,
            'train_r2': float(metrics['train_r2']),
            'val_mae': float(metrics['val_mae']),
            'val_tweedie': float(metrics['val_tweedie']),
            'n_features': len(features),
            'n_clusters': N_CLUSTERS,
        }
        pd.DataFrame([summary]).to_csv(OUTDIR / 'model_summary.csv', index=False)

        del x_train, x_val, y_train, y_val, test_feat
        gc.collect()
        print_memory_usage('final')

        print('
' + '=' * 70)
        print('🎉 GC_final_v2 완료!')
        print('=' * 70)
        print('
📂 출력:')
        print(f'  - {SUBMISSION_PATH} (제출 파일)')
        print(f'  - {OUTDIR / "prediction_scatter.png"}')
        print(f'  - {OUTDIR / "model_summary.csv"}')
        print(f'  - {OUTDIR / "feature_importance.csv"}')
        print(f'  - {OUTDIR / "feature_importance.png"}')
        print(f'  - {OUTDIR / "location_map.png"}')


    if __name__ == '__main__':
        main()
