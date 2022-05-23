import json


class Replacer:
    def __init__(self, table_path) -> None:
        self.table_path = table_path
        self.mapping = {}

    def pre(self, str_in, db_id):
        if (
            isinstance(str_in, str)
            and str_in.lower() in self.mapping[db_id]["orig2name"]
        ):
            str_in = self.mapping[db_id]["orig2name"][str_in]
        return str_in

    def post(self, str_in, db_id):
        if (
            isinstance(str_in, str)
            and str_in.lower() in self.mapping[db_id]["name2orig"]
        ):
            str_in = self.mapping[db_id]["name2orig"][str_in]
        return str_in
