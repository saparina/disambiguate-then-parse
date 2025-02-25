import re
import sqlite3
from collections import defaultdict

def check_and_modify_names(name):
    name = name.lower()
    if re.match(r'^\d', name) or "(" in name or "%" in name or name == "index" or name == "from" or name == "group" or name == "order" or name == "transaction" or name == "case" or name == "join" or "-" in name and '"' not in name:
        return f'"{name}"'
    elif " " in name:
        return name.replace(" ", "_")
    else:
        return name
    
def db_to_text(name):
    return name.replace("_", " ")

def dump_database_raw(db_path):
    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_path) as conn:
            dump_file = ""
            # Use the connection to generate the dump
            for line in conn.iterdump():
                dump_file += f"{line}\n"
    except Exception as e:
        print(f"Error creating database dump: {e}")
        import pdb; pdb.set_trace()
    return dump_file

def dump_database(db_path, synonyms, syn_type, make_different_values=False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    conn.text_factory = lambda b: b.decode(errors = 'ignore')

    cursor.execute("PRAGMA encoding;")
    cursor.fetchone()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    sql_dump = ""

    for table in tables:
        original_table_name = table[0].lower()

        if original_table_name == "sqlite_sequence":
            continue

        if syn_type == "table" and original_table_name in synonyms:
            table_names = synonyms[original_table_name]
        else:
            table_names = [original_table_name]

        tbl_col_types = {}
        
        for num_tbl, table_name in enumerate(table_names):

            cursor.execute(f"PRAGMA table_info({original_table_name})")
            columns = cursor.fetchall()

            create_table_parts = []
            original_column_names = []
            all_column_names = []
            primary_keys = []

            for col in columns:
                original_col_name = col[1].lower()
                original_column_names.append(original_col_name)

                if syn_type != "table" and original_table_name in synonyms and original_col_name in synonyms[original_table_name]:
                    col_names = synonyms[original_table_name][original_col_name]
                    all_column_names.extend(col_names)
                else:
                    all_column_names.append(check_and_modify_names(original_col_name))
                    col_names = [original_col_name]
                
                col_type = col[2]
                if original_table_name not in tbl_col_types:
                    tbl_col_types[original_table_name] = {}
                tbl_col_types[original_table_name][original_col_name] = col_type

                constraints = ""
                if col[3]:  # NOT NULL
                    constraints += " NOT NULL"
                if col[5]:  # PRIMARY KEY

                    if original_col_name not in primary_keys:
                        primary_keys += col_names
                    else:
                        primary_keys.append(check_and_modify_names(original_col_name))

                if col[4] is not None:  # Default value
                    constraints += f" DEFAULT {col[4]}"

                for col_name in col_names:
                    create_table_parts.append(f"    {check_and_modify_names(col_name)} {col_type}{constraints}")      

            if primary_keys:
                create_table_parts.append(f"    PRIMARY KEY ({', '.join(primary_keys)})")          

            create_table_sql = f"CREATE TABLE {check_and_modify_names(table_name)} (\n" + ",\n".join(create_table_parts) + "\n);"
            sql_dump += create_table_sql + "\n"

            # Insert data
            cursor.execute(f"SELECT * FROM {original_table_name}")
            rows = cursor.fetchall()
            for row in rows:
                insert_parts = [f"INSERT INTO {check_and_modify_names(table_name)} ("]
                insert_parts.append(", ".join(all_column_names))
                insert_parts.append(") VALUES (")
                
                values = []
                for orig_col, value in zip(original_column_names, row):
                    if isinstance(value, str):
                        value = value.replace("\n", " ")
                    if syn_type != "table" and original_table_name in synonyms and orig_col in synonyms[original_table_name]:
                        col_synonyms = synonyms[original_table_name][orig_col]
                        assert len(col_synonyms) == 2, col_synonyms
                        if make_different_values:
                            values.append(value)
                            col_type = tbl_col_types[original_table_name][original_col_name]
                            if value is None:
                                value = "" if 'text' in col_type or 'char' in col_type or 'clob' in col_type else 0
                            elif isinstance(value, str):
                                value += "_"
                            else:
                                value += 1
                            values.append(value)
                        else:
                            values.extend([value] * len(col_synonyms))

                    elif syn_type == "table" and original_table_name in synonyms and num_tbl == 1 and make_different_values:
                            col_type = tbl_col_types[original_table_name][original_col_name]
                            if value is None:
                                value = "" if 'text' in col_type or 'char' in col_type or 'clob' in col_type else 0
                            elif isinstance(value, str):
                                value += "_"
                            else:
                                value += 1
                            values.append(value)

                    else:
                        values.append(value)
                
                insert_parts.append(", ".join("?" for _ in values))
                insert_parts.append(");")
                insert_sql = "".join(insert_parts)
    

                cursor.execute("SELECT " + ", ".join(f"quote(?)" for _ in values), values)
                quoted_values = cursor.fetchone()
                sql_dump += insert_sql.replace("?", "{}").format(*quoted_values) + "\n"
            # sql_dump += "\n"

    conn.close()

    return sql_dump

def clean_column_name(col_name):
    # Remove quotes and extra spaces
    col_name = col_name.strip().replace('"', '').replace('\\', '')
    # Replace spaces with underscores
    col_name = col_name.replace(' ', '_')
    # Remove any other special characters
    col_name = ''.join(c for c in col_name if c.isalnum() or c == '_')
    return col_name.lower()

def dump_database_split(db_path, split_map, schema_with_content):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    conn.text_factory = lambda b: b.decode(errors = 'ignore')

    cursor.execute("PRAGMA encoding;")
    cursor.fetchone()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    sql_dump = ""

    # Process schema to get table definitions
    tables_schema = {}
    for table_def in schema_with_content.split(" | "):
        table_name, columns_str = table_def.strip().split(" : ")
        # Handle complex column definitions
        columns = []
        current_col = []
        in_quotes = False
        
        for char in columns_str:
            if char == '"' or char == '\\':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                columns.append(''.join(current_col).strip())
                current_col = []
            else:
                current_col.append(char)
        
        if current_col:  # Add the last column
            columns.append(''.join(current_col).strip())
            
        # Clean column names
        columns = [clean_column_name(col) for col in columns]
        tables_schema[table_name] = columns

    # First, handle existing tables
    for table in tables:
        original_table_name = table[0].lower()

        if original_table_name == "sqlite_sequence":
            continue

        # Get table structure
        cursor.execute(f"PRAGMA table_info({original_table_name})")
        columns = cursor.fetchall()

        create_table_parts = []
        column_names = []
        primary_keys = []

        for col in columns:
            col_name = check_and_modify_names(col[1].lower())
            column_names.append(col_name)
            
            col_type = col[2]
            constraints = ""
            if col[3]:  # NOT NULL
                constraints += " NOT NULL"
            if col[5]:  # PRIMARY KEY
                primary_keys.append(col_name)
            if col[4] is not None:  # Default value
                constraints += f" DEFAULT {col[4]}"
            
            create_table_parts.append(f"    {col_name} {col_type}{constraints}")

        if primary_keys:
            create_table_parts.append(f"    PRIMARY KEY ({', '.join(primary_keys)})")

        create_table_sql = f"CREATE TABLE {original_table_name} (\n" + ",\n".join(create_table_parts) + "\n);"
        sql_dump += create_table_sql + "\n"

        # Insert data for existing tables
        cursor.execute(f"SELECT * FROM {original_table_name}")
        rows = cursor.fetchall()
        for row in rows:
            insert_parts = [f"INSERT INTO {original_table_name} ("]
            insert_parts.append(", ".join(column_names))
            insert_parts.append(") VALUES (")
            
            # Clean up values, replacing newlines with spaces
            values = []
            for val in row:
                if isinstance(val, str):
                    val = val.replace('\n', ' ')
                values.append(val)
            
            insert_parts.append(", ".join("?" for _ in values))
            insert_parts.append(");")
            insert_sql = "".join(insert_parts)

            cursor.execute("SELECT " + ", ".join(f"quote(?)" for _ in values), values)
            quoted_values = cursor.fetchone()
            sql_dump += insert_sql.replace("?", "{}").format(*quoted_values) + "\n"

    # Now create the new split tables
    for table_name, split_column in split_map.items():
        if table_name in tables_schema:
            new_table_name = f"{table_name}_{split_column}"
            if new_table_name in tables_schema:
                columns = tables_schema[new_table_name]
                create_table_parts = []
                for col in columns:
                    create_table_parts.append(f"    {check_and_modify_names(col)} TEXT")
                
                create_table_sql = f"CREATE TABLE {new_table_name} (\n" + ",\n".join(create_table_parts) + "\n);"
                sql_dump += create_table_sql + "\n"

    conn.close()
    return sql_dump


def dump_database_agg(db_path, schema_without_content, all_raw_cols, all_cols, new_table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    conn.text_factory = lambda b: b.decode(errors = 'ignore')

    cursor.execute("PRAGMA encoding;")
    cursor.fetchone()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    sql_dump = ""

    # Process schema to get table definitions
    tables_schema = {}
    for table_def in schema_without_content.split(" | "):
        table_name, columns = table_def.strip().split(" : ")
        tables_schema[table_name] = columns.split(" , ")

    # First handle existing tables
    for table in tables:
        original_table_name = table[0].lower()

        if original_table_name == "sqlite_sequence":
            continue

        # Get table structure
        cursor.execute(f"PRAGMA table_info({original_table_name})")
        columns = cursor.fetchall()

        create_table_parts = []
        column_names = []
        primary_keys = []

        # Add existing columns
        for col in columns:
            col_name = check_and_modify_names(col[1].lower())
            column_names.append(col_name)
            
            col_type = col[2]
            constraints = ""
            if col[3]:  # NOT NULL
                constraints += " NOT NULL"
            if col[5]:  # PRIMARY KEY
                primary_keys.append(col_name)
            if col[4] is not None:  # Default value
                constraints += f" DEFAULT {col[4]}"
            
            create_table_parts.append(f"    {col_name} {col_type}{constraints}")

        if primary_keys:
            create_table_parts.append(f"    PRIMARY KEY ({', '.join(primary_keys)})")

        create_table_sql = f"CREATE TABLE {original_table_name} (\n" + ",\n".join(create_table_parts) + "\n);"
        sql_dump += create_table_sql + "\n"

        # Insert data for existing tables
        cursor.execute(f"SELECT * FROM {original_table_name}")
        rows = cursor.fetchall()
        for row in rows:
            insert_parts = [f"INSERT INTO {original_table_name} ("]
            insert_parts.append(", ".join(column_names))
            insert_parts.append(") VALUES (")
            
            values = []
            for val in row:
                if isinstance(val, str):
                    val = val.replace('\n', ' ')
                values.append(val)

            insert_parts.append(", ".join("?" for _ in values))
            insert_parts.append(");")
            insert_sql = "".join(insert_parts)

            cursor.execute("SELECT " + ", ".join(f"quote(?)" for _ in values), values)
            quoted_values = cursor.fetchone()
            sql_dump += insert_sql.replace("?", "{}").format(*quoted_values) + "\n"

    # Create the new aggregation table
    if new_table_name in tables_schema:
        create_table_parts = []
        for col in tables_schema[new_table_name]:
            col_type = "REAL" if col in all_cols else "TEXT"
            create_table_parts.append(f"    {check_and_modify_names(col)} {col_type}")
        
        create_table_sql = f"CREATE TABLE {new_table_name} (\n" + ",\n".join(create_table_parts) + "\n);"
        sql_dump += create_table_sql + "\n"

    conn.close()
    return sql_dump


def create_db(db_path, sql_dump):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Split the dump into individual SQL statements
    sql_statements = re.split(r';\s*\n', sql_dump)

    # Execute each SQL statement
    for statement in sql_statements:
        # Skip empty statements and comments
        if statement.strip() and not statement.strip().startswith('--'):
            try:
                cursor.execute(statement)
            except sqlite3.Error as e:
                print(f"Error executing statement: {statement}")
                print(f"Error message: {e}")
                import pdb; pdb.set_trace()
                raise RuntimeError

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def merge_all_insert_statements_ambiqt(db_dump):
    # Dictionary to store values for each table
    table_inserts = defaultdict(dict)

    new_content = ""

    # Process the dump line by line
    for line in db_dump.split('\n'):
        if "INSERT INTO " in line:
            table_name = line[line.find("INSERT INTO") + len("INSERT INTO"):line.find("(")].strip()
            col_names = line[line.find("(") + 1:line.find(") VALUES")]
            line = line[line.find("VALUES")+ len("VALUES"):-1]

            # Append values to the corresponding table's list
            if table_name == '"sqlite_sequence"':
                continue
            if col_names not in table_inserts[table_name]:
                table_inserts[table_name][col_names] = []
            table_inserts[table_name][col_names].append(line)
        elif "BEGIN TRANSACTION" in line or "COMMIT" in line or "DELETE" in line:
            continue
        else:
            new_content += line + "\n"

    # Create a single INSERT INTO statement for each table
    for table_name, col_names in table_inserts.items():
        for col_name, values in col_names.items():
            if values:
                new_content += f"INSERT INTO {table_name} ({col_name})" + " VALUES " + ",".join(values[:5]) + ";\n"

    return new_content

def filter_db_dump(db_dump, gold_ambig_queries):
    # filter dump
    db_dump_filtered = ''
    if isinstance(gold_ambig_queries, list):
        gold_ambig_queries = " ".join(gold_ambig_queries)
    all_queries_str = gold_ambig_queries.lower()
    skip = False
    for line in db_dump.split('\n'):
        if line.startswith("CREATE TABLE"):
            tab_name = line[len("CREATE TABLE "):line.find("(")].strip().lower()
            if tab_name.startswith('"') or tab_name.startswith("'"):
                tab_name = tab_name[1:-1]
            if tab_name in all_queries_str:
                db_dump_filtered += line + "\n"
                skip = False
            else:
                skip = True
        elif line.startswith("INSERT INTO"):
            skip = False
            tab_name = line[len("INSERT INTO "):line.find("(")].strip().lower()
            if tab_name.startswith('"') or tab_name.startswith("'"):
                tab_name = tab_name[1:-1]
            if tab_name in all_queries_str:
                db_dump_filtered += line + "\n"
        elif not skip:
            db_dump_filtered += line + "\n"
    return db_dump_filtered

def get_column_names(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"PRAGMA table_info({table_name});")
    rows = cursor.fetchall()
    column_names = [row[1] for row in rows]
    conn.close()
    return column_names

def merge_all_insert_statements(db_path, db_dump):
    # Dictionary to store values for each table
    table_inserts = defaultdict(list)

    new_content = ""

    # Process the dump line by line
    for line in db_dump.split('\n'):
        if "INSERT INTO " in line:
            table_name = line[line.find("INSERT INTO ") + len("INSERT INTO "):line.find(" VALUES(")]
            line = line[line.find("VALUES")+ len("VALUES"):-1]

            # Append values to the corresponding table's list
            if table_name != '"sqlite_sequence"':
                table_inserts[table_name].append(line)
        elif "BEGIN TRANSACTION" in line or "COMMIT" in line or "DELETE" in line:
            continue
        else:
            new_content += line + "\n"

    # Create a single INSERT INTO statement for each table
    for table_name, values in table_inserts.items():
        if values:
            col_names = get_column_names(db_path, table_name)
            new_content += f"INSERT INTO {table_name} (" + ",".join(col_names) + ") VALUES " + ",".join(values) + ";\n"

    return new_content

