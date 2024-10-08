import pandas as pd

# Učitajte CSV fajl u DataFrame
df = pd.read_csv('NIS_ESP_telemetry_anonymized.csv')  # Zamenite sa vašom putanjom do fajla

# Popunjavanje praznih vrednosti samo unutar iste bušotine (well)
df_filled = df.groupby('well', group_keys=False).apply(lambda group: group.fillna(method='ffill'))

# Export-ujte popunjeni DataFrame u novi CSV fajl
df_filled.to_csv('NIS_ESP_telemetry_filled.csv', index=False)  # Zamenite sa željenim imenom fajla

# Opcionalno možete pogledati rezultate
print("Podaci su uspešno exportovani u 'NIS_ESP_telemetry_filled.csv'")
