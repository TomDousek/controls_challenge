import pandas as pd

# Cesta k souboru
input_path = "./data/00000.csv"
output_path = "./data/00000_modified.csv"  # nebo stejné jméno pokud chceš přepsat

# Načtení dat
df = pd.read_csv(input_path)

# Změna všech hodnot road_roll na 0
df["roll"] = 0.0

# Uložení zpět do CSV
df.to_csv(output_path, index=False)

print("Hotovo. Soubor uložen jako:", output_path)

# import pprint
# obdoba vardump
# pprint.pprint(future_plan)