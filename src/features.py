import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder


    def add_time_and_interactions(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour.astype(np.int8)
        df['minute'] = df['time'].dt.minute.astype(np.int8)
        df['month'] = df['time'].dt.month.astype(np.int8)
        df['dayofyear'] = df['time'].dt.dayofyear.astype(np.int16)
        df['dayofweek'] = df['time'].dt.dayofweek.astype(np.int8)
        df['day'] = df['time'].dt.day.astype(np.int8)
        df['quarter'] = df['time'].dt.quarter.astype(np.int8)
        df['weekofyear'] = df['time'].dt.isocalendar().week.astype(np.int16)
        df['time_of_day_minutes'] = (df['hour'] * 60 + df['minute']).astype(np.int16)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)

        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(np.int8)
        df['sun_elevation_proxy'] = np.maximum(0, np.sin((df['hour'] - 6) * np.pi / 12)).astype(np.float32)

        if 'temp_b' in df.columns:
            df['temp_avg'] = df[['temp_a', 'temp_b']].mean(axis=1).astype(np.float32)
        else:
            df['temp_avg'] = df['temp_a'].astype(np.float32)

        if 'uv_idx' in df.columns and 'cloud_a' in df.columns:
            df['clearness'] = (100 - df['cloud_a'].fillna(50)).clip(0, 100).astype(np.float32) / 100
            df['uv_clearness_interaction'] = (df['uv_idx'].fillna(0) * df['clearness']).astype(np.float32)

        if 'cloud_a' in df.columns:
            cloud_norm = df['cloud_a'].fillna(50).clip(0, 100) / 100
            df['hour_cloud_interaction'] = (df['hour'] * cloud_norm).astype(np.float32)
            df['peak_cloud_penalty'] = (((df['hour'] >= 10) & (df['hour'] <= 14)).astype(np.int8) * cloud_norm).astype(np.float32)

        if 'humidity' in df.columns:
            df['dryness_index'] = ((100 - df['humidity'].fillna(50)) * np.maximum(0, df['temp_avg'].fillna(15)) / 100).astype(np.float32)

        if 'uv_idx' in df.columns:
            df['solar_uv_interaction'] = (df['sun_elevation_proxy'] * df['uv_idx'].fillna(0)).astype(np.float32)
            df['solar_uv_squared'] = (df['solar_uv_interaction'] ** 2).astype(np.float32)

        noon_distance = np.abs(12 - df['hour']) / 12
        df['temp_noon_interaction'] = (df['temp_avg'].fillna(15) * (1 - noon_distance)).astype(np.float32)

        if {'cloud_a', 'cloud_b'} <= set(df.columns):
            df['cloud_mean'] = df[['cloud_a', 'cloud_b']].mean(axis=1).astype(np.float32)
        elif 'cloud_a' in df.columns:
            df['cloud_mean'] = df['cloud_a'].astype(np.float32)

        if 'humidity' in df.columns:
            df['dryness'] = (100.0 - df['humidity']).astype(np.float32)

        if {'temp_max', 'temp_min'} <= set(df.columns):
            df['temp_range'] = (df['temp_max'] - df['temp_min']).astype(np.float32)

        if {'wind_spd_a', 'wind_spd_b'} <= set(df.columns):
            df['wind_spd_mean'] = df[['wind_spd_a', 'wind_spd_b']].mean(axis=1).astype(np.float32)

        if 'uv_idx' in df.columns and 'cloud_mean' in df.columns:
            df['uv_x_clear'] = (df['uv_idx'] * (100.0 - df['cloud_mean'])).astype(np.float32)

        return df


    def add_location_features(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_clusters: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
        print('
[위치 Feature Engineering]')
        print('=' * 50)

        df_train = df_train.copy()
        df_test = df_test.copy()
        df_train['__is_train__'] = 1
        df_test['__is_train__'] = 0
        df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        print(f"  coord1 범위: [{df_all['coord1'].min():.2f}, {df_all['coord1'].max():.2f}]")
        print(f"  coord2 범위: [{df_all['coord2'].min():.2f}, {df_all['coord2'].max():.2f}]")

        df_all['coord1_norm'] = (df_all['coord1'] - df_all['coord1'].min()) / (df_all['coord1'].max() - df_all['coord1'].min() + 1e-9)
        df_all['coord2_norm'] = (df_all['coord2'] - df_all['coord2'].min()) / (df_all['coord2'].max() - df_all['coord2'].min() + 1e-9)
        df_all['coord_distance'] = np.sqrt(df_all['coord1'] ** 2 + df_all['coord2'] ** 2).astype(np.float32)
        df_all['coord_angle'] = np.arctan2(df_all['coord2'], df_all['coord1']).astype(np.float32)
        df_all['coord1_x_coord2'] = (df_all['coord1'] * df_all['coord2']).astype(np.float32)
        df_all['coord1_plus_coord2'] = (df_all['coord1'] + df_all['coord2']).astype(np.float32)
        df_all['coord1_minus_coord2'] = (df_all['coord1'] - df_all['coord2']).astype(np.float32)

        est_lat = df_all['coord1'].astype(float).values
        if (np.nanmax(np.abs(est_lat)) < 5.0) or (np.nanstd(est_lat) < 0.5):
            used_lat = np.full_like(est_lat, 35.0, dtype=np.float32)
            print('  ⚠️ coord1이 위도로 부적절하여 위도 35°로 대체(보수적)')
        else:
            used_lat = np.clip(est_lat, -66.0, 66.0).astype(np.float32)
            print('  💡 coord1을 위도로 사용(클램핑 적용)')
        df_all['estimated_latitude'] = used_lat

        df_all['latitude_solar_factor'] = np.cos(np.radians(df_all['estimated_latitude'])).astype(np.float32)
        df_all['hour_x_latitude'] = (df_all['hour'] * df_all['estimated_latitude']).astype(np.float32)
        df_all['sun_elev_proxy_x_lat'] = (df_all['sun_elevation_proxy'] * df_all['latitude_solar_factor']).astype(np.float32)

        print(f'
  🎯 지리적 클러스터링 ({n_clusters}개)...')
        pv_coords = df_all.groupby('pv_id')[['coord1', 'coord2']].first().reset_index()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pv_coords['location_cluster'] = kmeans.fit_predict(pv_coords[['coord1', 'coord2']])

        df_all = df_all.merge(pv_coords[['pv_id', 'location_cluster']], on='pv_id', how='left')
        df_all['location_cluster'] = df_all['location_cluster'].astype(np.int8)

        cluster_centers = kmeans.cluster_centers_
        df_all['distance_to_cluster_center'] = 0.0
        for cluster_id in range(n_clusters):
            mask = df_all['location_cluster'] == cluster_id
            center = cluster_centers[cluster_id]
            distances = np.sqrt((df_all.loc[mask, 'coord1'] - center[0]) ** 2 + (df_all.loc[mask, 'coord2'] - center[1]) ** 2)
            df_all.loc[mask, 'distance_to_cluster_center'] = distances
        df_all['distance_to_cluster_center'] = df_all['distance_to_cluster_center'].astype(np.float32)

        print('
  ⭐ 클러스터별 최대 일사량 포텐셜 계산(0.98 quantile)...')
        train_only = df_all[df_all['__is_train__'] == 1]
        cluster_max_map = train_only.groupby('location_cluster')['nins'].quantile(0.98).to_dict()
        for cluster_id, value in sorted(cluster_max_map.items()):
            print(f'    Cluster {cluster_id}: {value:.1f} W/m²')

        df_all['cluster_max_potential'] = df_all['location_cluster'].map(cluster_max_map).fillna(650).astype(np.float32)
        df_all['hour_max_ratio'] = (df_all['sun_elevation_proxy'] * df_all['cluster_max_potential']).astype(np.float32)
        df_all['cluster_potential_normalized'] = (df_all['cluster_max_potential'] / 750.0).astype(np.float32)

        train_out = df_all[df_all['__is_train__'] == 1].drop(columns=['__is_train__']).reset_index(drop=True)
        test_out = df_all[df_all['__is_train__'] == 0].drop(columns=['__is_train__']).reset_index(drop=True)
        return train_out, test_out, pv_coords, cluster_centers


    def add_solar_geometry_approx(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lat = df['estimated_latitude'].astype(float).values
        doy = df['dayofyear'].astype(int).values
        gamma = 2.0 * np.pi * (doy - 1) / 365.0
        declination = (
            0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma)
            + 0.00148 * np.sin(3 * gamma)
        ).astype(np.float32)

        hour = df['hour'].astype(float).values
        minute = df['minute'].astype(float).values
        hour_angle = (hour + minute / 60.0 - 12.0) * 15.0 * np.pi / 180.0
        latitude_radians = np.radians(lat)

        sin_elev = (
            np.sin(latitude_radians) * np.sin(declination)
            + np.cos(latitude_radians) * np.cos(declination) * np.cos(hour_angle)
        )
        elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0))).astype(np.float32)
        df['solar_elev_deg'] = elevation
        return df


    def encode_identifiers(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = train.copy()
        test = test.copy()

        le_pv = LabelEncoder()
        le_pv.fit(pd.concat([train['pv_id'].astype(str), test['pv_id'].astype(str)], ignore_index=True))
        train['pv_id_encoded'] = le_pv.transform(train['pv_id'].astype(str)).astype(np.int32)
        test['pv_id_encoded'] = le_pv.transform(test['pv_id'].astype(str)).astype(np.int32)

        if 'type' in train.columns:
            le_type = LabelEncoder()
            le_type.fit(
                pd.concat(
                    [
                        train['type'].fillna('unknown').astype(str),
                        test['type'].fillna('unknown').astype(str),
                    ],
                    ignore_index=True,
                )
            )
            train['type_encoded'] = le_type.transform(train['type'].fillna('unknown').astype(str)).astype(np.int16)
            test['type_encoded'] = le_type.transform(test['type'].fillna('unknown').astype(str)).astype(np.int16)

        return train, test


    def select_base_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
        exclude = ['time', 'pv_id', 'nins', 'energy']
        if 'type' in train.columns:
            exclude.append('type')

        features = sorted((set(train.columns) & set(test.columns)) - set(exclude))
        return [col for col in features if not col.endswith('_original')]
