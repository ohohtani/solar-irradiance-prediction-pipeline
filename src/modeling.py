import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit

    from .config import (
        EARLY_STOPPING_ROUNDS,
        LEARNING_RATE,
        LOG_PERIOD,
        N_ESTIMATORS,
        TWEEDIE_POWER,
    )


    def pv_te_from(
        train_pv: pd.Series,
        train_y: pd.Series,
        apply_pv: pd.Series,
        global_mean: float,
    ) -> pd.Series:
        te_map = pd.DataFrame({'pv_id': train_pv.values, 'y': train_y.values}).groupby('pv_id')['y'].mean()
        return apply_pv.map(te_map).fillna(global_mean).astype(np.float32)


    def build_train_valid_test(
        train: pd.DataFrame,
        test: pd.DataFrame,
        features_base: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, set, set]:
        x_all = train[features_base].copy()
        y_all = train['nins'].clip(lower=0).astype(np.float32)

        print('
[pv_id Group Split 8:2]')
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        tr_idx, va_idx = next(splitter.split(x_all, y_all, groups=train['pv_id']))

        x_train = x_all.iloc[tr_idx].copy()
        x_val = x_all.iloc[va_idx].copy()
        y_train = y_all.iloc[tr_idx].copy()
        y_val = y_all.iloc[va_idx].copy()

        pv_tr = set(train.iloc[tr_idx]['pv_id'].unique())
        pv_va = set(train.iloc[va_idx]['pv_id'].unique())
        print(f'  Train pv_ids: {len(pv_tr)}, Valid pv_ids: {len(pv_va)}, Overlap: {len(pv_tr & pv_va)}')

        global_mean = float(y_train.mean())
        x_train['pv_te'] = pv_te_from(train.iloc[tr_idx]['pv_id'], y_train, train.iloc[tr_idx]['pv_id'], global_mean)
        x_val['pv_te'] = pv_te_from(train.iloc[tr_idx]['pv_id'], y_train, train.iloc[va_idx]['pv_id'], global_mean)

        test_feat = test[features_base].copy()
        test_feat['pv_te'] = pv_te_from(train.iloc[tr_idx]['pv_id'], y_train, test['pv_id'], global_mean)

        return x_train, x_val, y_train, y_val, test_feat, pv_tr, pv_va


    def train_lightgbm(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series):
        print('
' + '=' * 70)
        print(f'🚀 Training (Tweedie p={TWEEDIE_POWER}, Group 8:2 split)')
        print('=' * 70)

        model = LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=TWEEDIE_POWER,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=-1,
            num_leaves=511,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_samples=15,
            min_split_gain=0.001,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric=['l1', 'tweedie'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=LOG_PERIOD),
            ],
        )
        return model


    def evaluate_model(model, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> dict:
        best_iter = model.best_iteration_
        train_pred = np.maximum(0, model.predict(x_train, num_iteration=best_iter))
        val_pred = np.maximum(0, model.predict(x_val, num_iteration=best_iter))

        train_r2 = r2_score(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        val_tweedie = np.nan
        if hasattr(model, 'best_score_') and isinstance(model.best_score_, dict):
            valid_scores = model.best_score_.get('valid_1', {})
            if isinstance(valid_scores, dict) and 'tweedie' in valid_scores:
                val_tweedie = valid_scores['tweedie']

        metrics = {
            'best_iteration': int(best_iter),
            'train_r2': float(train_r2),
            'val_mae': float(val_mae),
            'val_tweedie': float(val_tweedie),
            'val_pred': val_pred,
            'train_pred': train_pred,
        }

        print('
' + '=' * 70)
        print('📊 RESULTS')
        print('=' * 70)
        print(f"best_iteration     : {metrics['best_iteration']}")
        print(f"train_acc (R2)     : {metrics['train_r2']:.6f}")
        print(f"val_loss (tweedie) : {metrics['val_tweedie']:.6f}")
        print(f"MAE (validation)   : {metrics['val_mae']:.6f}")
        print('※ pv_id 그룹 분리 → 새 발전소 일반화 추정에 유리')
        print('=' * 70)

        return metrics
