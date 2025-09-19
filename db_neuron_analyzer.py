import sqlite3
import os
import sys

# Database paths
source_db = r"C:\CrestDB\CREST_browsing_database_goog14r0s5c3_eirepredict2023.db"
output_db = "neuron_synapses_new.db"

# Define what we consider "actual neurons/cells" (not fragments)
# Based on database analysis - these are complete cell types, not fragments
NEURON_TYPES = {
    'pyramidal neuron',
    'interneuron', 
    'excitatory/spiny neuron with atypical tree',
    'spiny stellate neuron',
    'unclassified neuron',
    'astrocyte',
    'oligodendrocyte', 
    'microglia/opc',
    'unknown cell',
    'c-shaped cell',
    'blood vessel cell'
}

def create_smaller_db():
    """Just copy synapses involving neurons to make a smaller database"""
    
    print("Creating smaller database with only neuron synapses...")
    print(f"Source: {source_db}")
    print(f"Output: {output_db}")
    
    # Remove existing output
    if os.path.exists(output_db):
        try:
            os.remove(output_db)
            print("Removed existing output database")
        except PermissionError:
            # File is locked, rename it instead
            backup_name = f"{output_db}.backup"
            if os.path.exists(backup_name):
                os.remove(backup_name)
            os.rename(output_db, backup_name)
            print(f"Renamed existing database to {backup_name}")
    
    # Create neuron types string for SQL
    neuron_types_str = "'" + "', '".join(NEURON_TYPES) + "'"
    
    # Connect to output database
    output_conn = sqlite3.connect(output_db)
    output_cursor = output_conn.cursor()
    
    try:
        # Attach source database
        print("Attaching source database...")
        sys.stdout.flush()
        output_cursor.execute(f"ATTACH DATABASE '{source_db}' AS source")
        
        # Copy only synapses involving neurons
        print("Copying synapses involving neurons...")
        sys.stdout.flush()
        
        create_and_copy = f"""
        CREATE TABLE edge_list_table AS
        SELECT * FROM source.edge_list_table 
        WHERE pre_type IN ({neuron_types_str}) OR post_type IN ({neuron_types_str})
        """
        output_cursor.execute(create_and_copy)
        
        # Get count
        output_cursor.execute("SELECT COUNT(*) FROM edge_list_table")
        copied_synapses = output_cursor.fetchone()[0]
        print(f"Copied {copied_synapses:,} synapses involving neurons")
        sys.stdout.flush()
        
        # Add basic indexes
        print("Creating indexes...")
        sys.stdout.flush()
        output_cursor.execute("CREATE INDEX idx_pre_type ON edge_list_table(pre_type)")
        output_cursor.execute("CREATE INDEX idx_post_type ON edge_list_table(post_type)")
        
        # Commit and detach
        output_conn.commit()
        output_cursor.execute("DETACH DATABASE source")
        
        # Get file sizes
        source_size = os.path.getsize(source_db) / (1024**3)  # GB
        output_size = os.path.getsize(output_db) / (1024**3)  # GB
        
        print(f"\nDone!")
        print(f"Source: {source_size:.2f} GB")
        print(f"Output: {output_size:.2f} GB") 
        print(f"Size reduction: {((source_size - output_size) / source_size * 100):.1f}%")
        print(f"Output file: {output_db}")
        
        # Show what types we captured
        print(f"\nAnalyzing captured cell types...")
        output_cursor.execute(f"""
            SELECT pre_type, COUNT(*) as count 
            FROM edge_list_table 
            WHERE pre_type IN ({neuron_types_str})
            GROUP BY pre_type 
            ORDER BY count DESC
        """)
        
        print("Pre-synaptic cell types captured:")
        for cell_type, count in output_cursor.fetchall():
            print(f"  {cell_type}: {count:,}")
            
        output_cursor.execute(f"""
            SELECT post_type, COUNT(*) as count 
            FROM edge_list_table 
            WHERE post_type IN ({neuron_types_str})
            GROUP BY post_type 
            ORDER BY count DESC
        """)
        
        print("\nPost-synaptic cell types captured:")
        for cell_type, count in output_cursor.fetchall():
            print(f"  {cell_type}: {count:,}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        output_conn.close()

if __name__ == "__main__":
    print("Creating smaller neuron database...")
    print("=" * 40)
    
    if not os.path.exists(source_db):
        print(f"Error: Source database not found at {source_db}")
        exit(1)
    
    create_smaller_db()
    print("\nDone!")