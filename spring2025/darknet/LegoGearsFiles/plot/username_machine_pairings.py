def get_machine(username):
    machine_map = {
        "akumik": "HiPerGator (UF)",
        "thakur.s": "HiPerGator (UF)",
        "julian.dominguez": "HiPerGator (UF)",
        "pnh2pj": "Afton (UVA)",
        "bdu4xa": "Afton (UVA)",
        "ukx5fv": "Afton (UVA)",
        "jpf": "MALTLab",
        "julian": "MALTLab"
    }
    return machine_map.get(username, "Personal Machine")
