import psycopg2
from functions import db_config
import sys, getopt
import pandas as pd
import pandas.io.sql as sqlio

def connect_db():
    """ Connect to the PostgreSQL database server """
    
    con = None
    try:
        
        # connect to the PostgreSQL server
        
        con = psycopg2.connect(host=db_config.server,
                                database=db_config.database,
                                user=db_config.database,
                                password=db_config.password)
        # create a cursor
        return (con)
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        print ('Could not connect to DB. Exiting..')
        sys.exit(2)


def parse_sql(sql_file):

    print('Parsing the SQL file')
    f = open(sql_file, "r")

    # Read the SQL file with the commands
    cmd_text = f.read()
    # Parse and extract the SQL statements. Statements are separated by ';'
    cmds = cmd_text.split(";")
    cmds = cmds[:-1]
    print ('\t Found {} SQL statements'.format(len(cmds)))
    return(cmds)
    


def execute_sql(cmds):

    print('Starting execution of SQL Statements')

    try:
        # Connect to the database
        con = connect_db()
        cur = con.cursor()

        # Execute the commands one by one
        for command in cmds:
            cur.execute(command)

        # Close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        con.commit()
        print('\t Execution successful. Commited all changes')

    except (Exception, psycopg2.DatabaseError) as error:
                print(error)
    finally:
        if con is not None:
            con.close()


def report_db_tables():
    print('Checking the tables in DB')
    try:
        # Connect to the database
        con = connect_db()
        cur = con.cursor()

        # Get the list of tables in the DB
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", ('public',))
        recs = cur.fetchall()
        print('\tFound {} tables'.format(len(recs)))
        
        for rec in recs:
            print('\t\t {}'.format(str(rec)))

        # Close communication with the PostgreSQL database server
        cur.close()


    except (Exception, psycopg2.DatabaseError) as error:
                print(error)
    finally:
        if con is not None:
            con.close()
            

def report_table_recs(tables):
    
    #Connect to the database
    con = connect_db()
    cur = con.cursor()

    try:
        for table in tables:
            qry = "SELECT COUNT(*) FROM %s" % table
            cur.execute(qry)
            cnt = cur.fetchall()
            cnt_rec = cnt[0][0]
            print ('\t Table {} has {} records'.format(table,cnt_rec))
            
        
        # Close communication with the PostgreSQL database server
        cur.close()
        
    except (Exception, psycopg2.DatabaseError) as error:
                print(error)
    finally:
        if con is not None:
            con.close()
            

def read_insert_data_into_tables(tables,datafiles):
    con = connect_db()
    cur = con.cursor()
                
    for table,datafile in zip(tables,datafiles):
        f = open(datafile, 'r')
        cur.copy_from(f, table, sep=',',null="")
        con.commit()
        print('\t Committed data into table: {}'.format(table))
        
    con.close()


def create_drop_table(argv):

    try:
        opts, args = getopt.getopt(argv,"i:", ["sqlfile="])
    except getopt.GetoptError:
	    print ('Usage: python create_drop_tables.py --sqlfile=<sql_file_path>')
	    sys.exit(2)

    req_options = 0
    for opt, arg in opts:
        if opt == '--sqlfile':
            sql_file = arg
            req_options = 1

    if (req_options == 0):
        print ('Usage: python create_db.py --sqlfile=<sql_file_path>')
        sys.exit(2)


    # Parse the SQL statements from the file
    cmds = parse_sql(sql_file)

    #Execute the SQLs
    execute_sql(cmds)

    #Report DB tables
    report_db_tables()

def upload_csv_to_db(file_path, db_table):
    con = connect_db()
    cur = con.cursor()
                
    
    f = open(file_path, 'r')
    cur.copy_from(f, db_table, sep=',',null="")
    con.commit()
    print('\t Committed data into table: {}'.format(db_table))
        
    con.close()


def execute_sql_on_db():
    con = connect_db()
    cur = con.cursor()
    
    sql = "select * from video_frame_vjump_pose;"
    cur.execute(sql)
    
    records = cur.fetchall()
    
    videofileName = []
    frame_no = []
    model_name = []
    model_version =  []
    detected_pose = []
    detected_pose_conf = []
    actual_pose = []
    
    for record in records:
        videofileName.append(record[0])
        frame_no.append(record[1])
        model_name.append(record[2])
        model_version.append(record[3])
        detected_pose.append(record[4])
        detected_pose_conf.append(record[5])
        actual_pose.append(record[6])
        
        df = pd.DataFrame ({'videofilename':videofileName,'frame_no':frame_no, 'model_name':model_name, 'model_version':model_version, 'detected_pose':detected_pose,'detected_pose_conf':detected_pose_conf, 'actual_pose':actual_pose})
    
    print("Found {} records in database".format(df.shape[0]))
    return df
  
def get_critical_frames(video_file, model_name, model_version, conf_threshold):
    con = connect_db()
    cur = con.cursor()
    
    sql = "select frame_no, detected_pose from video_frame_vjump_pose where videofilename= '{}' AND model_name = '{}' AND model_version = '{}' AND detected_pose_conf > {};".format(video_file, model_name,  model_version, conf_threshold)
    cur.execute(sql)
    
    records = cur.fetchall()
    frame_no = []
    detected_pose = []
    detected_pose_conf = []
    
    for record in records:
        frame_no.append(record[0])
        detected_pose.append(record[1])

        df = pd.DataFrame ({'frame_no':frame_no, 'detected_pose':detected_pose})
    
    print("Found {} records in database".format(df.shape[0]))
    return df
    


if __name__ == "__main__":
   create_drop_table(sys.argv[1:])