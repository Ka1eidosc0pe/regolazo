"""Fictional, culturally varied name pools (first names, surnames, ISO 3166 alpha-3 code) and club identities.

Names are combined only within a region so they read plausibly; ``FAMOUS`` blocks combinations that belong to
well-known real players.
"""

from typing import Final

REGIONS: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "NGA": (
        ("Chidi", "Emeka", "Tunde", "Obinna", "Kelechi", "Ifeanyi", "Segun", "Uche", "Nnamdi", "Dayo"),
        ("Okafor", "Adeyemi", "Nwosu", "Balogun", "Eze", "Okonkwo", "Ogunleye", "Chukwu", "Ibe", "Adewale"),
    ),
    "GHA": (
        ("Kwame", "Kofi", "Yaw", "Kwabena", "Kojo", "Fiifi", "Nana"),
        ("Mensah", "Asante", "Owusu", "Boateng", "Appiah", "Darko", "Agyei", "Quaye"),
    ),
    "SEN": (
        ("Moussa", "Ibrahima", "Cheikh", "Abdoulaye", "Pape", "Lamine", "Babacar"),
        ("Diallo", "Ndiaye", "Sarr", "Faye", "Diop", "Gueye", "Seck", "Cissé"),
    ),
    "CIV": (
        ("Wilfried", "Serge", "Yannick", "Junior", "Arsène", "Ismaël"),
        ("Koné", "Bamba", "Kouassi", "Gnagbo", "Yao", "Aka", "Tiéné"),
    ),
    "BRA": (
        ("João", "Rafael", "Mateus", "Caio", "Felipe", "Vinícius", "Guilherme", "Davi", "Henrique"),
        ("Santos", "Oliveira", "Souza", "Pereira", "Carvalho", "Ribeiro", "Almeida", "Barbosa", "Teixeira"),
    ),
    "ARG": (
        ("Facundo", "Joaquín", "Nicolás", "Santiago", "Agustín", "Tomás", "Franco", "Ignacio"),
        ("Gómez", "Romero", "Acosta", "Medina", "Herrera", "Sosa", "Benítez", "Ojeda", "Ferreyra"),
    ),
    "ESP": (
        ("Álvaro", "Javier", "Sergio", "Pablo", "Iker", "Unai", "Marcos", "Rubén", "Adrián"),
        ("García", "Ruiz", "Navarro", "Moreno", "Ortega", "Castillo", "Ferrán", "Iglesias", "Vidal"),
    ),
    "FRA": (
        ("Théo", "Hugo", "Mathis", "Bastien", "Antoine", "Yanis", "Clément", "Maxence", "Quentin"),
        ("Lefèvre", "Moreau", "Girard", "Bonnet", "Rousseau", "Lambert", "Fournier", "Garnier", "Chevalier"),
    ),
    "ENG": (
        ("Jack", "George", "Oliver", "Callum", "Reece", "Mason", "Alfie", "Joe", "Ollie", "Ryan"),
        ("Walker", "Hughes", "Turner", "Fletcher", "Barnes", "Whitaker", "Holloway", "Pickering", "Ashworth"),
    ),
    "GER": (
        ("Lukas", "Jonas", "Felix", "Niklas", "Moritz", "Tobias", "Jannik", "Leon"),
        ("Becker", "Hoffmann", "Schäfer", "Krüger", "Brandt", "Vogel", "Lehmann", "Winkler", "Dietrich"),
    ),
    "NED": (
        ("Daan", "Sem", "Jesse", "Bram", "Ruben", "Thijs", "Stijn", "Joost"),
        ("de Vries", "Bakker", "Visser", "Smit", "Mulder", "Kuipers", "Hendriks", "van Dijkhuis"),
    ),
    "ITA": (
        ("Lorenzo", "Matteo", "Davide", "Alessandro", "Federico", "Riccardo", "Tommaso"),
        ("Rossi", "Marchetti", "Bianchi", "Ferrari", "Conti", "Romano", "Galli", "Fabbri", "Monti"),
    ),
    "POR": (
        ("Diogo", "Gonçalo", "Tiago", "Nuno", "Rodrigo", "Duarte"),
        ("Ferreira", "Rodrigues", "Gomes", "Martins", "Lopes", "Sousa", "Pinheiro", "Carvalhal"),
    ),
    "CRO": (
        ("Luka", "Ivan", "Marko", "Ante", "Josip", "Domagoj", "Filip"),
        ("Kovač", "Horvat", "Jurić", "Babić", "Marić", "Novak", "Šimić", "Perković"),
    ),
    "SRB": (
        ("Nikola", "Stefan", "Miloš", "Dušan", "Vuk", "Lazar"),
        ("Petrović", "Jovanović", "Nikolić", "Ilić", "Stanković", "Đorđević", "Marković"),
    ),
    "POL": (
        ("Jakub", "Kacper", "Mateusz", "Bartosz", "Szymon", "Wiktor"),
        ("Nowak", "Kowalski", "Wiśniewski", "Zieliński", "Kamiński", "Wójcik", "Dąbrowski"),
    ),
    "SWE": (
        ("Erik", "Oskar", "Viktor", "Emil", "Linus", "Albin"),
        ("Lindqvist", "Johansson", "Nyström", "Berglund", "Sandberg", "Ekström", "Holmgren"),
    ),
    "NOR": (
        ("Magnus", "Sander", "Henrik", "Aksel", "Eirik", "Torstein"),
        ("Halvorsen", "Berg", "Solberg", "Haugen", "Dahl", "Strand", "Bakken"),
    ),
    "DEN": (
        ("Mads", "Rasmus", "Frederik", "Mikkel", "Asger"),
        ("Andersen", "Kjær", "Mortensen", "Lund", "Thygesen", "Vestergaard"),
    ),
    "JPN": (
        ("Haruto", "Ren", "Sota", "Yuto", "Kaito", "Takumi", "Riku", "Hayato"),
        ("Tanaka", "Suzuki", "Takahashi", "Watanabe", "Ito", "Nakamura", "Kobayashi", "Fujimoto", "Hayashi"),
    ),
    "KOR": (
        ("Min-jun", "Seo-jun", "Ji-ho", "Do-yun", "Hyun-woo", "Jae-won", "Tae-yang"),
        ("Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Yoon", "Han"),
    ),
    "MAR": (
        ("Youssef", "Hakim", "Amine", "Karim", "Omar", "Bilal", "Zakaria", "Ilyas"),
        ("Benali", "El Idrissi", "Haddad", "Mansour", "Bensaïd", "Ziani", "Tahiri", "Alaoui"),
    ),
    "EGY": (
        ("Ahmed", "Mostafa", "Karim", "Tarek", "Hossam", "Ziad"),
        ("Fathy", "Hegazy", "Abdelaziz", "Shehata", "Ramadan", "Gaber"),
    ),
    "TUR": (
        ("Emre", "Burak", "Kerem", "Arda", "Mert", "Efe", "Yusuf"),
        ("Yılmaz", "Demir", "Kaya", "Çelik", "Aydın", "Öztürk", "Arslan", "Doğan"),  # noqa: RUF001
    ),
    "MEX": (
        ("Diego", "Andrés", "Emiliano", "Rodrigo", "Ulises", "Héctor"),
        ("Hernández", "Ramírez", "Castro", "Vargas", "Morales", "Ochoa", "Salcedo", "Treviño"),
    ),
    "COL": (
        ("Camilo", "Sebastián", "Jhon", "Yerson", "Duván", "Andrés"),
        ("Rojas", "Cuadrado", "Mosquera", "Valencia", "Murillo", "Palacios", "Cardona"),
    ),
    "USA": (
        ("Tyler", "Brandon", "Cole", "Ethan", "Caleb", "Owen", "Miles"),
        ("Mitchell", "Carter", "Brooks", "Sullivan", "Hayes", "Bennett", "Coleman", "Reed"),
    ),
    "IRL": (
        ("Ciarán", "Seán", "Ruairí", "Oisín", "Darragh", "Cillian"),
        ("Byrne", "Murphy", "Kelly", "Doyle", "Gallagher", "Brennan", "Quinlan"),
    ),
    "SCO": (
        ("Fraser", "Euan", "Lachlan", "Callum", "Ross", "Hamish"),
        ("MacLeod", "Ferguson", "Sinclair", "Munro", "Drummond", "Buchanan"),
    ),
    "AUS": (
        ("Riley", "Jayden", "Lachie", "Cooper", "Mitchell"),
        ("O'Brien", "McKay", "Kennedy", "Fitzgerald", "Harrison", "Lowe"),
    ),
    "IRN": (
        ("Arash", "Dariush", "Kian", "Reza", "Omid", "Navid"),
        ("Rahimi", "Karimi", "Hosseini", "Moradi", "Jafari", "Sadeghi"),
    ),
    "GEO": (
        ("Giorgi", "Levan", "Luka", "Saba", "Nika"),
        ("Beridze", "Kapanadze", "Gelashvili", "Lomidze", "Tsiklauri"),
    ),
    "CMR": (
        ("Bertrand", "Christian", "Jean-Pierre", "Landry", "Olivier"),
        ("Mbarga", "Ngadeu", "Onana-Bela", "Ekambi-Ndo", "Tchamba", "Fotso"),
    ),
}

FAMOUS: Final = frozenset(
    {
        "Luka Kovač",
        "Ivan Rakitić",
        "Mateus Santos",
        "Felipe Santos",
        "Jack Walker",
        "Kyle Walker",
        "Hakim Ziani",
        "Lee Kang",
        "Kim Lee",
        "Park Ji",
        "Sergio García",
        "Pablo García",
        "Javier García",
        "Diego Hernández",
        "Rafael Santos",
    }
)

CITIES: Final = (
    "Harbour City",
    "Ashford Vale",
    "Port Meridian",
    "Kestrel Bay",
    "Eastmoor",
    "Redcliffe",
    "Silverford",
    "Northgate",
    "Lowmere",
    "Brackenridge",
    "Copperfield",
    "Westhaven",
    "Thornbury Cross",
    "Saltmarsh",
    "Highcombe",
    "Ravensholm",
    "Duncastle",
    "Marlow Heath",
    "Ember Valley",
    "Stonebridge",
    "Glenwick",
    "Fairhollow",
    "Oakhurst",
    "Lunar Bay",
)

CLUB_SUFFIXES: Final = (
    "FC",
    "Athletic",
    "Rovers",
    "Wanderers",
    "Mariners",
    "Comets",
    "Harriers",
    "Pilots",
    "Lanterns",
    "Forge",
    "Tide",
    "Kestrels",
    "Sporting",
    "Town",
)

VENUE_WORDS: Final = (
    "Lantern Park",
    "The Foundry",
    "Meridian Ground",
    "Kestrel Stadium",
    "Riverside Bowl",
    "Beacon Road",
    "The Harbour Arena",
    "Copper Lane",
    "Northlight Stadium",
    "Ironworks Park",
)

KIT_COLOURS: Final = (
    "#0b3d91",  # navy
    "#c8102e",  # red
    "#f2f4f8",  # white
    "#ffc72c",  # gold
    "#00843d",  # green
    "#6a1b9a",  # purple
    "#ff6f00",  # orange
    "#1e88e5",  # sky blue
    "#212121",  # black
    "#7b1e3a",  # claret
    "#00a19a",  # teal
    "#d81b60",  # magenta
)

GK_COLOURS: Final = ("#2dc653", "#ff9f1c", "#f15bb5", "#c6ff00", "#00e5ff", "#ffd60a")
