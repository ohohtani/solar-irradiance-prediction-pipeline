from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


    def plot_prediction_scatter(model, x_val: pd.DataFrame, y_val: pd.Series, best_iter: int, outdir: Path) -> None:
        print('
🎨 예측 산점도 생성 중...')
        val_pred = np.maximum(0, model.predict(x_val, num_iteration=best_iter))

        if len(y_val) > 50000:
            idx = np.random.choice(len(y_val), 50000, replace=False)
            y_sample = y_val.iloc[idx].values
            pred_sample = val_pred[idx]
        else:
            y_sample = y_val.values
            pred_sample = val_pred

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(y_sample, pred_sample, alpha=0.3, s=5, c='steelblue')
        axes[0].plot([0, 1300], [0, 1300], 'r--', linewidth=2, label='Perfect')
        axes[0].set_xlabel('Actual nins (W/m²)', fontsize=12)
        axes[0].set_ylabel('Predicted nins (W/m²)', fontsize=12)
        axes[0].set_title('Actual vs Predicted (Validation)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1300)
        axes[0].set_ylim(0, 1300)

        residuals = y_sample - pred_sample
        axes[1].hist(residuals, bins=100, alpha=0.7, color='coral', edgecolor='black')
        axes[1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
        axes[1].set_xlabel('Residual (Actual - Predicted) W/m²', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        scatter_png = outdir / 'prediction_scatter.png'
        plt.savefig(scatter_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  ✅ prediction_scatter.png -> {scatter_png}')


    def save_feature_importance(model, features: list[str], outdir: Path) -> pd.DataFrame:
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(outdir / 'feature_importance.csv', index=False)
        return importance_df


    def plot_feature_importance(importance_df: pd.DataFrame, features: list[str], outdir: Path) -> None:
        location_features = [
            feature for feature in features
            if any(token in feature for token in ['coord', 'location', 'latitude', 'cluster', 'distance', 'hour_x_latitude', 'sun_elev_proxy_x_lat'])
        ]

        print('
[시각화 생성 중...]')
        fig, ax = plt.subplots(figsize=(10, 12))
        top_30 = importance_df.head(30)
        colors = ['red' if feat in location_features else 'steelblue' for feat in top_30['feature']]

        ax.barh(range(len(top_30)), top_30['importance'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels(top_30['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(
            handles=[
                Patch(facecolor='red', alpha=0.7, label='Location'),
                Patch(facecolor='steelblue', alpha=0.7, label='Other'),
            ],
            loc='lower right',
        )

        plt.tight_layout()
        plt.savefig(outdir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ feature_importance.png -> {outdir / 'feature_importance.png'}")


    def plot_location_map(
        pv_coords_info: pd.DataFrame,
        cluster_centers: np.ndarray,
        pv_tr: set,
        pv_va: set,
        outdir: Path,
    ) -> None:
        if pv_coords_info is None or cluster_centers is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        train_pv_coords = pv_coords_info[pv_coords_info['pv_id'].isin(list(pv_tr))]
        val_pv_coords = pv_coords_info[pv_coords_info['pv_id'].isin(list(pv_va))]

        sc1 = axes[0].scatter(
            train_pv_coords['coord1'],
            train_pv_coords['coord2'],
            c=train_pv_coords['location_cluster'],
            cmap='tab10',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
        )
        axes[0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centers')
        axes[0].set_title(f'Train Plants (n={len(train_pv_coords)})', fontweight='bold')
        axes[0].set_xlabel('coord1')
        axes[0].set_ylabel('coord2')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        plt.colorbar(sc1, ax=axes[0], label='Cluster')

        sc2 = axes[1].scatter(
            val_pv_coords['coord1'],
            val_pv_coords['coord2'],
            c=val_pv_coords['location_cluster'],
            cmap='tab10',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
        )
        axes[1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centers')
        axes[1].set_title(f'Valid Plants (n={len(val_pv_coords)})', fontweight='bold')
        axes[1].set_xlabel('coord1')
        axes[1].set_ylabel('coord2')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        plt.colorbar(sc2, ax=axes[1], label='Cluster')

        plt.tight_layout()
        plt.savefig(outdir / 'location_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ location_map.png -> {outdir / 'location_map.png'}")
