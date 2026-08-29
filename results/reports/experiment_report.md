# Smart Solar IoT Monitoring System - Experiment Report

Generated: 2026-08-28T05:16:31.917518

## Configuration
- Duration: 1 days
- Sampling interval: 5 min
- Random seed: 42
- Scenarios: 9

## Summary Statistics
       repetition  experiment_id  total_energy_wh  mean_power_w  max_power_w  mean_efficiency_pct  capacity_factor  performance_ratio  iot_latency_ms  iot_packet_loss_pct  iot_delivery_rate_pct  iot_fault_detection_rate  traditional_data_availability_pct  traditional_energy_error_pct  traditional_fault_detection_rate  traditional_response_time_s  power_error_improvement_pct  energy_error_improvement_pct
count    27.00000      27.000000        27.000000     27.000000    27.000000            27.000000        27.000000          27.000000       27.000000            27.000000              27.000000                      27.0                       2.700000e+01                     27.000000                         27.000000                         27.0                    27.000000                     27.000000
mean      1.00000       5.000000      1884.590394     78.712768   259.044659            26.761057         0.196782           3.025891       50.925362             4.031568              95.968432                     100.0                       8.333333e+00                      1.727855                          0.770315                       3600.0                    98.201124                     99.999996
std       0.83205       2.631174       587.208534     24.611216    69.089654             3.614633         0.061528           0.631463        0.597667             2.589261               2.589261                       0.0                       1.810195e-15                      2.270926                          2.232302                          0.0                     0.739420                      0.000004
min       0.00000       1.000000       847.577101     35.315713   137.196718            19.484892         0.088289           1.994350       50.195691             1.736111              88.541667                     100.0                       8.333333e+00                      0.047434                          0.000000                       3600.0                    96.353703                     99.999983
25%       0.00000       3.000000      1348.300498     56.179187   195.880041            23.216158         0.140448           2.323782       50.431209             1.751448              95.138889                     100.0                       8.333333e+00                      0.546962                          0.000000                       3600.0                    97.724554                     99.999995
50%       1.00000       5.000000      2259.460772     94.326856   297.276987            27.470752         0.235817           3.264956       50.592351             3.125000              96.875000                     100.0                       8.333333e+00                      0.798916                          0.000000                       3600.0                    98.681405                     99.999998
75%       2.00000       7.000000      2351.400221     98.562801   313.959014            30.362553         0.246407           3.522315       51.670630             4.861111              98.248552                     100.0                       8.333333e+00                      1.905064                          0.000000                       3600.0                    98.739789                     99.999998
max       2.00000       9.000000      2568.856388    107.035683   337.872287            31.510625         0.267589           3.775379       52.120991            11.458333              98.263889                     100.0                       8.333333e+00                     10.248601                          7.894737                       3600.0                    98.848357                     99.999999

## PV Performance by Scenario
 total_energy_wh  mean_power_w  max_power_w  mean_efficiency_pct  capacity_factor  energy_yield_wh_wp  performance_ratio              scenario
     2566.949262    106.956219   336.212226            22.285534         0.267391            6.417373           2.128666      normal_clear_sky
      850.303875     35.429328   147.799602            23.956815         0.088573            2.125760           2.314281        low_irradiance
     2380.932558     99.205523   309.178802            21.593589         0.248014            5.952331           2.097485      high_temperature
     1631.758148     67.989923   251.042996            24.477763         0.169975            4.079395           2.471689       cloud_variation
     2263.844546     94.326856   297.276987            21.622721         0.235817            5.659611           2.096980     dust_accumulation
     1300.115387     54.171474   172.106926            19.484892         0.135429            3.250288           1.994350       partial_shading
     2254.042106     95.577757   310.286694            22.475500         0.238944            5.635105           2.155091        sensor_failure
     2325.845876     96.910245   310.286694            22.374491         0.242276            5.814615           2.129966 communication_failure
     1367.786385     56.991099   203.168934            22.312483         0.142478            3.419466           2.333283      mixed_conditions

## Sensor Calibration
               sensor  unit      mae      rmse      bias      mape
  BH1750 (irradiance) W/m^2 9.469748 13.453159  3.695809 73.351736
DS18B20 (temperature)    °C 0.408312  0.513017 -0.007433  1.311360
   PZEM-017 (voltage)     V 0.160329  0.201193 -0.001192  0.505673
   PZEM-017 (current)     A 0.029672  0.042734  0.008852  3.788609
     PZEM-017 (power)     W 1.226421  1.757031  0.408535 13.161434

## ML Model Comparison (Power Prediction)
            model      mae     rmse       r2 status
linear_regression 2.012140 2.519000 0.999679     ok
    random_forest 1.989518 2.780854 0.999608     ok
gradient_boosting 1.985653 2.740607 0.999620     ok
          xgboost 2.236931 3.119519 0.999507     ok

## Fault Detection Summary
- normal: 864

## Traditional Monitoring Metrics
- data_availability_pct: 8.333333333333332
- energy_estimation_error_pct: 0.6442387706467068
- fault_detection_rate_pct: 0.0
- response_time_s: 3600
- sampling_interval_min: 60
- total_samples: 72