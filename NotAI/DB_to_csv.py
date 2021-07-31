# -*- coding:utf-8 -*-
from cx_Oracle import connect
from datetime import datetime

class DB2csv:
    def __init__(self, db='choi', db_name='choi', ip='localhost'):
        self.db, self.db_name, self.ip = db, db_name, ip
    
    def score(self):
        try:
            con = connect('%s/%s@%s:1521/xe' % (self.db, self.db_name, self.ip))  # db에 연결
            cur = con.cursor()
            
            sql = """
            select ng_no, max(nc_no)-min(nc_no)+1, max(nc_score), max(nc_level), max(nc_line) -- 게임 번호, 진행 프레임 수, 최종 점수, 최종 레벨, 최종 라인수
            from NotAI_game3, NotAI_Control3
            where ng_no = nc_ng_no
            group by ng_no
            order by ng_no
            """
            cur.execute(sql)
            
            try:
                now = datetime.strftime(datetime.today(), '%Y%m%d-%H%M%S')
                f = open(r'C:\orbeat\NotAI\data\score_%s.csv' % now, 'w', encoding='utf-8')
            except Exception as e:
                print('파일 열기 실패 :', e)
            
            a_score = 0
            for no, frame, score, level, line in cur:
                a_score += score
                f.write('%d,%d,%d,%d,%d,%d\n' % (no, frame, score, a_score, level, line) )
                
            try:
                f.close()
            except Exception as e:
                print('파일 닫기 실패 :', e)
                
                
            
            
        except Exception as e:
            print('DB불러오기 실패1', e)
        
        try:
            con.close()
        except Exception as e:
            print('DB연결 해제 실패', e)
            
to_csv = DB2csv()
to_csv.score()
