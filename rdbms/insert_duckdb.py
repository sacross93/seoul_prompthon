import duckdb
from glob import glob

conn = duckdb.connect("./seoul_prompthon/rdbms/prompthon.db")
cursor = conn.cursor()


### create table
cursor.execute("create sequence parsing_serial start 1;")
cursor.execute("drop sequence parsing_serial;")
# cursor.execute("ALTER SEQUENCE parsing_serial RESTART WITH 1;")

cursor.execute("""
               create table parsing_list (
                   id integer primary key default nextval('parsing_serial'),
                   source varchar(255) not null,
                   data_path varchar(255) not null,
                   process boolean not null default 0,
                   process_date timestamp default current_timestamp,
                   update_count integer default 0
                )
               """)

cursor.execute("""drop table parsing_list;""")


tables = cursor.execute("show tables;")
tables.fetchall()

### example insert
import duckdb
import pandas as pd

seouls = pd.read_excel('./seoul_prompthon/seoul_promp.xlsx')

conn = duckdb.connect("./seoul_prompthon/rdbms/prompthon.db")
cursor = conn.cursor()

cursor.execute("""
              INSERT INTO parsing_list (source, data_path) 
              VALUES ('https://sehub.net/archives/2088351', './seoul_prompthon/pdf/동대문구 ESG경제 허브센터 신규 입주기업 모집 재공고.pdf');
              """)

cursor.execute("""
               INSERT INTO parsing_list (source, data_path)
               VALUES ('https://edu.seoulsbdc.or.kr/course/course_view.jsp?id=49942&ch=course1', './seoul_prompthon/figures/‘어르신 돌봄도 스마트하게’… 서울시, 로봇·AI로 어르신 건강·안전지킨다_page1_img2.png');
               """)

cursor.execute("""
               select *
               from parsing_list;
               """)
print(cursor.fetchall())

### test

# df_list = pd.DataFrame(
#     {
#         "source":
#             [
#                 "https://www.seoulsbdc.or.kr/bs/BS_VIEW.do?currentPage=1&boardCd=B061&infoSeq=132&bbs_Viewcnt=0&rnoIndex=undefined&searchType=title&searchStr=&viewcnt=10", "https://www.seoulsbdc.or.kr/bs/BS_VIEW.do?currentPage=1&boardCd=B061&infoSeq=132&bbs_Viewcnt=0&rnoIndex=undefined&searchType=title&searchStr=&viewcnt=10", "https://www.seoulsbdc.or.kr/bs/BS_VIEW.do?currentPage=1&boardCd=B061&infoSeq=109&bbs_Viewcnt=0&rnoIndex=undefined&searchType=title&searchStr=&viewcnt=10", "https://www.seoulsbdc.or.kr/bs/BS_VIEW.do?currentPage=1&boardCd=B061&infoSeq=109&bbs_Viewcnt=0&rnoIndex=undefined&searchType=title&searchStr=&viewcnt=10", "https://www.seoulsbdc.or.kr/bs/BS_VIEW.do?currentPage=1&boardCd=B061&infoSeq=102&bbs_Viewcnt=0&rnoIndex=undefined&searchType=title&searchStr=&viewcnt=10"
#                 ], 
#         "data_path":
#             [
#                 "중대재해처벌법 대응 경영컨설팅 지원사업 참여기업 모집(서울경제인협회).html", "'Recruitment of companies participating in the Serious Accident Punishment Act Corporate Consulting Support Project (Seoul Business Association).png'", "[중구] 2024년 중구 맞춤형 소상공인 지원사업 상반기 모집공고.html", "'[Jung-gu] Recruitment notice for the first half of 2024 Jung-gu customized small business support project.png'", "'[Jung-gu] Recruitment notice for the first half of 2024 Jung-gu customized small business support project.pdf'"
#                 ]
#             }
#     )
# df_list.to_excel("./seoul_prompthon/seoul_promp_test.xlsx", index=False)