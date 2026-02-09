#!/usr/bin/env python3
"""
Add WBC 2026 rosters to teams_2026.json by matching pitcher names to MLBAM IDs.
Uses pitcher_seasons.json as the name→ID lookup.
"""
import json
import re

# WBC 2026 pitcher rosters (names as they appear in MLB records)
WBC_PITCHERS = {
    "WBC-USA": [
        "Boyd, Matt", "Cleavinger, Garrett", "Kershaw, Clayton", "Skubal, Tarik", "Speier, Gabe",
        "Bednar, David", "Holmes, Clay", "Jax, Griffin", "Keller, Brad", "McLean, Nolan",
        "Miller, Mason", "Ryan, Joe", "Skenes, Paul", "Wacha, Michael", "Webb, Logan", "Whitlock, Garrett"
    ],
    "WBC-JPN": [
        "Kikuchi, Yusei", "Matsui, Yuki", "Yamamoto, Yoshinobu",
        "Sugano, Tomoyuki"
    ],
    "WBC-DOM": [
        "Peralta, Wandy", "Sanchez, Cristopher", "Soto, Gregory",
        "Alcantara, Sandy", "Bello, Brayan", "Brazoban, Huascar",
        "Dominguez, Seranthony", "Doval, Camilo", "Estevez, Carlos",
        "Santana, Dennis", "Severino, Luis", "Uceta, Edwin", "Uribe, Abner", "Abreu, Albert",
        "Alvarado, Elvis"
    ],
    "WBC-VEN": [
        "Alvarado, Jose", "Rodriguez, Eduardo", "Suarez, Ranger", "Zerpa, Angel",
        "Lopez, Pablo", "Marquez, German", "Montero, Keider", "Senzatela, Antonio",
        "Palencia, Daniel", "De Jesus, Enmanuel", "Gomez, Yoendrys", "Machado, Andres",
        "Bazardo, Eduard", "Butto, Jose", "Sanchez, Ricardo", "Mosqueda, Oddanier", "Guzman, Carlos"
    ],
    "WBC-PR": [
        "Moran, Jovani", "Rivera, Eduardo",
        "Cruz, Fernando", "De Leon, Jose", "Diaz, Edwin", "Garcia, Rico",
        "Lopez, Jorge", "Lugo, Seth", "Burgos, Raymond", "Rodriguez, Gabriel",
        "Espada, Jose", "Quinones, Luis", "Reyes, Angel", "Rios, Yacksel",
        "Rodriguez, Elmer", "Velez, Ricardo"
    ],
    "WBC-MEX": [
        "Assad, Javier", "Bradley, Taj", "Munoz, Andres", "Walker, Taijuan",
        "Vodnik, Victor", "Duarte, Daniel", "Reyes, Gerardo", "Urquidy, Jose",
        "Bernardino, Brennan", "Garcia, Robert", "Armenta, Alexander",
        "Cruz, Jesus", "Gastelum, Luis", "Carrillo, Alex", "Natera, Samy"
    ],
    "WBC-CAN": [
        "Allen, Logan", "Paxton, James", "Quantrill, Cal", "Taillon, Jameson",
        "Soroka, Michael", "Balazovic, Jordan", "Macko, Adam",
        "Jean, Antoine", "Wilkinson, Matt", "Zastryzny, Rob",
        "Aumont, Phillippe", "Cerantola, Eric", "Diaz, Indigo", "Loewen, Carter", "Skirrow, Noah",
        "Ashman, Micah"
    ],
    "WBC-KOR": [
        "Ryu, Hyun Jin", "Dunning, Dane", "O'Brien, Riley"
    ],
    "WBC-ITA": [
        "Nola, Aaron", "Lorenzen, Michael", "Marinaccio, Ron", "Ottavino, Adam",
        "Festa, Matt", "Graceffo, Gordon", "Nicolas, Kyle", "Weissert, Greg",
        "Altavilla, Dan", "DeLucia, Dylan", "Jacob, Alek", "Aldegheri, Sam",
        "La Sorsa, Joe", "Ercolani, Alessandro", "Quattrini, Gabriele"
    ],
    "WBC-CUB": [
        "Moinelo, Livan", "Robaina, Julio", "Rodriguez, Yariel", "Rodriguez, Osiel",
        "Martinez, Raiden", "Cruz, Naykel", "Nunez, Darien",
        "Lopez, Yoan", "Alvarez, Frank", "Chapman, Emmanuel",
        "Cousin, Josimar", "Larrondo, Denny", "Romero, Luis", "Santos, Pedro",
        "Hurtado, Daviel", "Martinez, Randy"
    ],
    "WBC-COL": [
        "Quintana, Jose", "Sanmartin, Reiver", "Patino, Luis", "Crismatt, Nabil",
        "Teheran, Julio", "Guerrero, Tayron", "Gomez, Rio",
        "Almeida, Adrian", "Bergner, Austin", "Correa, Danis",
        "Garcia, Pedro", "Lorduy, David", "Martinez, Emerson", "Romero, Jhon",
        "Gomez, Yapson", "Alcala, Elkin", "Zuniga, Guillo"
    ],
    "WBC-NED": [
        "Jansen, Kenley", "Kelly, Kevin",
        "Floranus, Wendell", "Huijer, Lars", "Mendez, Eric", "West, Derek",
        "Wilson, Dylan", "Martis, Shairon", "Sulbaran, JC",
        "Cornelia, Jamdrick", "Merite, Ryjeteri", "Croes, Dayson",
        "Estanista, Jaydenn", "Fransen, Arij", "Kelly, Antwone",
        "Kelly, Jaitoine", "Morales, Justin", "Oduber, Shawndrick"
    ],
    "WBC-GBR": [
        "Beck, Tristan", "Beck, Brendan", "Fernander, Chavez", "Knowles, Antonio",
        "Worley, Vance", "Benoit, Donovan",
        "Wells, Nick", "Wylie, Justin", "Anderson, Jack",
        "Hill, Gary Gill", "Langhorne, Miles", "Long, Ryan",
        "Petersen, Michael", "Seppings, Jack", "Spraker, Graham",
        "Victor, Najer", "Viza, Tyler", "Wild, Owen"
    ],
    "WBC-TPE": [
        "Hu, Chih-Wei", "Lin, Yu Min"
    ],
    "WBC-PAN": [
        "Allen, Logan", "Barria, Jaime", "Espino, Paolo", "Jurado, Ariel",
        "Mejia, Humberto", "Ramirez, JC", "Agrazal, Dario",
        "Baldonado, Alberto", "Cienfuegos, Miguel", "Gonzalez, James",
        "Hernandez, Kenny", "Otero, Andy", "Garcia, Jorge",
        "Gomez, Miguel", "Guerra, Javy", "Mendoza, Abdiel", "Rodriguez, Erian"
    ],
    "WBC-NIC": [
        "Ramirez, Erasmo", "Ramirez, JC", "Rodriguez, Carlos",
        "Torres, Bryan", "Bermudez, Danilo", "Mejia, Dilmer",
        "Rayo, Oscar", "Teller, Carlos", "Burton, Kenword",
        "Cruz, Stiven", "Gutierrez, Osman", "Hebbert, Duque",
        "Medrano, Ronald", "Obando, Angel"
    ],
    "WBC-AUS": [
        "Wells, Alex", "Saupold, Warwick",
        "Hendrickson, Josh", "Kennedy, Jon", "Morgan, Cooper",
        "O'Loughlin, Jack", "Townsend, Blake", "Wells, Lachlan",
        "Hall, Kieren", "Hampton, Ky", "Holland, Sam",
        "MacDonald, Connor", "Neunborn, Mitch", "Van Steensel, Todd", "Wynne, Coen"
    ],
    "WBC-BRA": [
        "Pardinho, Eric", "Vieira, Thyago",
        "Kanabushi, Hugo", "Nakaoshi, Oscar", "Sawayama, Enzo",
        "Villarroel, Hector", "Albanez, Pietro", "Barbosa, Gabriel",
        "Contreras, Joseph", "Da Silva, Tiago", "Gouvea, Murilo",
        "Lemos, Pedro", "Lopez, Tomas", "Missaki, Daniel", "Takahashi, Rodrigo"
    ],
    "WBC-CZE": [
        "Barto, Jeff", "Duffek, Tomas", "Ercoli, Lukas", "Novak, Jan",
        "Capka, Filip", "Hlouch, Lukas", "Kollmann, Filip",
        "Kovala, Michael", "Minarik, Marek", "Ondra, Tomas",
        "Padysak, Daniel", "Satoria, Ondrej", "Schneider, Martin",
        "Vank, Ondrej", "Vecerka, Boris"
    ]
}

# Team display names
WBC_TEAM_NAMES = {
    "WBC-USA": "United States",
    "WBC-JPN": "Japan",
    "WBC-DOM": "Dominican Republic",
    "WBC-VEN": "Venezuela",
    "WBC-PR":  "Puerto Rico",
    "WBC-MEX": "Mexico",
    "WBC-CAN": "Canada",
    "WBC-KOR": "Korea",
    "WBC-ITA": "Italy",
    "WBC-CUB": "Cuba",
    "WBC-COL": "Colombia",
    "WBC-NED": "Netherlands",
    "WBC-GBR": "Great Britain",
    "WBC-TPE": "Chinese Taipei",
    "WBC-PAN": "Panama",
    "WBC-NIC": "Nicaragua",
    "WBC-AUS": "Australia",
    "WBC-BRA": "Brazil",
    "WBC-CZE": "Czechia",
}

def normalize(name):
    """Normalize a name for matching — lowercase, strip accents/punctuation."""
    name = name.lower().strip()
    # Remove common accent chars
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ñ': 'n', 'ü': 'u', 'ö': 'o', 'ä': 'a',
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    # Remove periods, apostrophes
    name = re.sub(r"[.'`]", "", name)
    return name

def main():
    # Build name→ID lookup from pitcher_seasons.json
    ps_path = "/Users/jackmorello/Desktop/MLB_PitcherChart/frontend/public/pitcher_seasons.json"
    with open(ps_path) as f:
        ps_data = json.load(f)

    # Use latest season's entry for each pitcher
    name_to_id = {}
    for r in ps_data:
        norm = normalize(r["player_name"])
        name_to_id[norm] = r["pitcher"]

    print(f"Loaded {len(name_to_id)} unique pitcher names from pitcher_seasons.json")

    # Load existing teams_2026.json
    teams_path = "/Users/jackmorello/Desktop/MLB_PitcherChart/frontend/public/teams_2026.json"
    with open(teams_path) as f:
        teams_data = json.load(f)

    total_matched = 0
    total_missed = 0

    for team_abbr, pitcher_names in WBC_PITCHERS.items():
        matched_ids = []
        missed_names = []

        for name in pitcher_names:
            norm = normalize(name)
            if norm in name_to_id:
                matched_ids.append(name_to_id[norm])
            else:
                # Try partial matching (last name only)
                last = norm.split(",")[0].strip()
                found = False
                for k, v in name_to_id.items():
                    if k.startswith(last + ","):
                        # Check if first name initial matches
                        first_wbc = norm.split(",")[-1].strip()[:3]
                        first_db = k.split(",")[-1].strip()[:3]
                        if first_wbc == first_db:
                            matched_ids.append(v)
                            found = True
                            break
                if not found:
                    missed_names.append(name)

        # Add to teams_2026.json
        display_name = WBC_TEAM_NAMES.get(team_abbr, team_abbr)
        teams_data["teams"][team_abbr] = {"name": display_name, "lg": "WBC"}
        teams_data["rosters"][team_abbr] = matched_ids

        total_matched += len(matched_ids)
        total_missed += len(missed_names)

        print(f"  {team_abbr:8s} {display_name:25s} → {len(matched_ids):2d}/{len(pitcher_names):2d} matched")
        if missed_names:
            print(f"           Missed: {', '.join(missed_names)}")

    print(f"\nWBC Total: {total_matched} matched, {total_missed} not in data")

    # Write updated file
    with open(teams_path, "w") as f:
        json.dump(teams_data, f, indent=2)
    print(f"Updated {teams_path}")

if __name__ == "__main__":
    main()
