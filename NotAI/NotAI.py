from PIL import ImageGrab
import PIL.Image as pilimg
from time import perf_counter as clock, sleep
import pygetwindow as gw
import numpy as np
import os
from PIL import Image
from datetime import datetime
from random import choice, sample, random
import _pyautogui_win as platformModule
from threading import Thread
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import cv2
if __name__ == '__main__': import TrainNotTetris as tnt
try:
    from cx_Oracle import connect
except Exception as e:
    print(e)


class Operation:

    def __init__(self, db='choi', db_name='choi', ip='localhost'):
        clock()
        self.db, self.db_name, self.ip = db, db_name, ip
        # self.x1, self.y1, self.x2, self.y2 = 17, 88, 98, 169#3, 26, 98, 169
        self.x1, self.y1, self.x2, self.y2 = 17, 26, 98, 169#3, 26, 98, 169
        
        font = np.array(pilimg.open(r'img\font.png'))  # 글씨체 이미지를 읽어옴
        self.number_font = []
        self.number_name = []
        for i in range(11):
            x1, y1 = 1 + 8 * i, 0
            x2, y2 = x1 + 7, y1 + 8
            self.number_font.append(np.mean(font[y1:y2, x1:x2, 0:3], axis=2))
            self.number_name.append(i)
        self.number_name[-1] = 0
        self.number_font = np.array(self.number_font)
        self.number_name = np.array(self.number_name)
        # print(self.number_font[0])
        # print(np.mean(number_font[0][:,:,0:3], axis=2))
        
        # 블럭 이미지를 불러옴
        self.blocks_img = []
        for i in range(1, 8):
            self.blocks_img.append(np.array(pilimg.open(r'img\pieces\%d.png' % i)))
            # print(self.blocks_img[i-1])
        
        print('블럭 데이터 준비중...')
        mino = 'IJLOSTZ'
        self.block_data = []
        self.block_data_label = []
        targetdir = None
        self.POOLING_X = 5
        self.POOLING_Y = 5
        for i, v1 in enumerate(mino):
            targetdir = r'img\rotation\%s' % v1
            files = os.listdir(targetdir)
            # print(files)
            for _, v in enumerate(files):
                # print(targetdir + '\\' + v)
                self.block_data.append(_pooling(self.POOLING_X, self.POOLING_Y,
                                                np.array(pilimg.open(r'img\rotation\%s\%s' % (v1, v)))[:,:,:3]))
                self.block_data_label.append(v1)
                
        self.block_data = np.array(self.block_data)
        print(self.block_data.shape)
        
        self.block_data_label = np.array(self.block_data_label)
        
        # print(data_label)
        
        # 블럭 이미지의 배경 색(255,255,255)을 제외한 RGB별 평균 수치를 구함
        for i in range(7):
            # if i==0: print(self.blocks_img[i][:,:,:3]!=np.array([255,255,255]))
            # if i==0: print(np.any(self.blocks_img[i][:,:,:3]!=np.array([255,255,255]), axis=2))
            # if i==0: print(self.blocks_img[i][np.any(self.blocks_img[i][:,:,:3]!=np.array([255,255,255]), axis=2)])
            self.blocks_img[i] = self.avg_RGB(self.blocks_img[i])
        
        self.blocks_img = np.array(self.blocks_img)[:,:3]
        print(self.blocks_img)
        
        self.key_li = [['', 'z', 'x'], ['', 'left', 'right'], ['', 'down']]  # 조작키 리스트(각각의 리스트에서는 한개의 키만 선택됨)
        # self.key_li = [['z', 'x'], ['left', 'right'], ['down']] # 조작키 리스트(각각의 리스트에서는 한개의 키만 선택됨)
        combination = 1  # 경우의 수(조합)계산
        for i in self.key_li:
            combination *= len(i)
        print(combination)
        self.nbActions = combination
        
        self.number2key_li = []
        for i in range(combination):
            self.number2key_li.append(self.number2key_bool_li(combination, i))
            # print(i)
            # print(self.number2key_bool_li(combination, i))

        # for i in self.number2key_li:
            # print(i)
        # exit()
        self.key_li2 = []
        for i in self.key_li:
            for j in i:
                if j == '': continue
                self.key_li2.append(j)
        
        self.push_t = np.random.normal(0.5, 1, 1000)
        self.push_t = self.push_t[(self.push_t > 0) & (self.push_t < 100)]  # 조작키를 누르는 시간(정규분포(0이하의 값과 100을 초과하는 값들은 버림))
        
        self.windows = None
        self.full_screenshot = None
        self.check_lobby = None
        self.score, self.level, self.line = None, None, None
        self.next_piece = None
        self.info_li = []
        
        self.saver = tf.train.Saver()
        self.game_no = self.current_game_no()
    
    def number2key_bool_li(self, combination, number):
        push_key = []
        # n = 0
        for i in self.key_li:
            combination //= len(i)  # 해당 자리의 값을 구함
            n = number // combination
            # print(combination, number, i, n)
            push_key.append(i[n])
            number -= n * combination
        
        bool_li = []
        for i in self.key_li:
            for j in i:
                if j == '': continue
                bool_li.append(j in push_key)
        
        return bool_li
    
    def key_bool_li2number(self, combination, key_bool_li):
        jali_li = []
        for i in self.key_li:
            combination //= len(i)  # 해당 자리의 값을 구함
            for j in range(len(i)):
                if j == 0: continue
                jali_li.append(j * combination)
        jali_li = np.array(jali_li)
        # print(jali_li)
        # print(key_bool_li)
        # print(jali_li[np.array(key_bool_li)==1])
            
        return np.sum(jali_li[np.array(key_bool_li) == 1])
    
    def avg_RGB(self, img):  # 해당 이미지에서 완전한 흰색(배경)을 제외한 RGB값의 평균을 반환함
        img = img[np.any(img[:,:,:3] != np.array([255, 255, 255]), axis=2)]  # 배경색([255,255,255]) 제외
        return np.mean(img, axis=0)  # 배경을 제외한 RGB값의 평균을 반환
    
    def check_score(self):
        score = 0
        num = None
        x1, y1, x2, y2 = None, None, None, None
        for i in range(6):
            x1, y1 = 147 - 8 * i, 50
            x2, y2 = x1 + 7, y1 + 8
            avg = np.mean(self.full_screenshot[y1:y2, x1:x2], axis=2)  # 스크린샷의 각 픽셀별 평균
            bo = avg == self.number_font  # 숫자 이미지의 픽셀값과 정확하게 일치하는 픽셀(각각의 숫자 이미지에 대한 값을 모두 구함)
            bo = np.all(bo, axis=2)  # 해당 줄의 픽셀값이 전부 같으면 True
            bo = np.all(bo, axis=1)  # 해당 열의 픽셀값이 전부 같으면 True
            num = self.number_name[bo][0]  # 최종적으로 픽셀값이 완전히 일치하는 숫자 이미지를 찾아냄
            score += 10 ** i * num  # 점수 더하기
        return score
    
    def check_level(self):
        level = 0
        num = None
        x1, y1, x2, y2 = None, None, None, None
        for i in range(2):
            x1, y1 = 139 - 8 * i, 82
            x2, y2 = x1 + 7, y1 + 8
            num = self.number_name[np.all(np.all(np.mean(self.full_screenshot[y1:y2, x1:x2], axis=2) == self.number_font, axis=2), axis=1)][0]
            level += 10 ** i * num
        return level
    
    def check_line(self):
        line = 0
        num = None
        x1, y1, x2, y2 = None, None, None, None
        for i in range(3):
            x1, y1 = 139 - 8 * i, 106
            x2, y2 = x1 + 7, y1 + 8
            num = self.number_name[np.all(np.all(np.mean(self.full_screenshot[y1:y2, x1:x2], axis=2) == self.number_font, axis=2), axis=1)][0]
            line += 10 ** i * num
        return line
    
    def check_next_piece(self):
        print(self.avg_RGB(self.full_screenshot[129:162, 122:155]))
            
        # if len(self.avg_RGB(self.full_screenshot[129:162, 122:155]))<=10: # 배경색이 아닌 부분이 10픽셀 이하면
            # return None # None값을 반환함
        
        av = self.avg_RGB(self.full_screenshot[129:162, 122:155])
        print(av, av[0], type(av[0]))
        cha = np.abs(self.blocks_img - av)
        # print(cha)
        hap = np.min(cha, axis=1)  # np.sum(cha, axis=1)
        print(hap)
        bo = hap == np.min(hap)
        return  np.arange(1, 8)[bo]
    
    def check_block(self):
        pool_img = _pooling(self.POOLING_X, self.POOLING_Y, self.full_screenshot[129:162, 122:155])
        
        cha = np.abs(self.block_data - pool_img)
        # print(cha.shape)
        hap = np.sum(np.sum(np.sum(cha, axis=3), axis=2), axis=1)
        bo = hap == np.min(hap)
        return self.block_data_label[bo]
    
    def current_game_no(self):
        try:
            con = connect('%s/%s@%s:1521/xe' % (self.db, self.db_name, self.ip))  # db에 연결
            cur = con.cursor()
            sql = """
            select max(ng_no) from NotAI_game3
            """
            # print(sql)
            cur.execute(sql)
            ng_no = 0
            for i in cur:
                # print(i)
                ng_no = i[0]
            if ng_no == None: ng_no = 0
        except:
            ng_no = 0
            
        return ng_no
    
    def save_data(self):
        try:
            con = connect('%s/%s@%s:1521/xe' % (self.db, self.db_name, self.ip))  # db에 연결
            cur = con.cursor()
            
            sql = """
            insert into NotAI_game3
            values (NotAI_game3_seq.nextval, to_date('%s', 'YYYYMMDD-HH24MISS'), %s, to_date('%s', 'YYYYMMDD-HH24MISS'), %s)
            """ % (self.start_game_time, self.start_game_clock, self.end_game_time, self.end_game_clock)
            # print(sql)
            cur.execute(sql)
            
            con.commit()  # 실제로 DB서버에 반영
            
            # 가장 최근에 진행한 게임 번호를 받아옴
            sql = """
            select max(ng_no) from NotAI_game3
            """
            # print(sql)
            cur.execute(sql)
            ng_no = None
            for i in cur:
                ng_no = i[0]
            # exit()
                  
            # con.close()
        except Exception as e:
            print('DB저장 실패', e)
        
        _dir = r'\orbeat\NotAI\data\img\%s_%.4f' % (self.start_game_time, self.start_game_clock)
        createFolder(_dir)
        
        #          게임 시작 시간             _시작 클럭      조작 시작 시각        _조작 시간                
        # data\img\yyyymmdd-himiss_clock()\clock()(%.6f)_push_t(%.6f)_조작키.png
        # _dir = None
        _path = None
        # 추후 OracleDB에 바로 저장하도록 바꾸기(스크린샷은 그대로 폴더에 저장)
        
        try:
            createFolder(r'\orbeat\NotAI\data\log')
            f = open(r'\orbeat\NotAI\data\log\%s_%.4f.txt' % (self.start_game_time, self.start_game_clock), 'a', encoding='UTF-8')
            for i, v in enumerate(self.info_li):
                _path = _dir + r'\%.4f_%s.png' % (v['current_clock'], v['key'])  # 이미지의 경로
                Image.fromarray(v['screenshot'], 'RGB').save(_path)
                
                f.write('%.4f\t%s\t%d\t%d\t%d\t%s\n' % (v['current_clock'], v['key'], v['score'], v['level'], v['line'], v['next_piece']))
                
                try:
                    # print(v)
                    sql = """
                    insert into NotAI_Control3
                    values (NotAI_Control3_seq.nextval, %.4f, %d, %d, %d, %d, %d, %d, %d, %d, '%s', %d)
                    """ % (v['current_clock'], v['key'][0], v['key'][1], v['key'][2], v['key'][3], v['key'][4],
                           v['score'], v['level'], v['line'], v['next_piece'], ng_no)
                    # print(sql)
                    cur.execute(sql)
                except Exception as e:
                    print('DB 저장 실패 :', e)
                
            f.close()
            
            try:
                con.commit()  # 실제로 DB서버에 반영          
                con.close()
            except Exception as e:
                print('commit error :', e)
        except Exception as e:
            print('저장 실패 :', e)
        
        return ng_no
    
    def jiyeon(self):
        self.game_time = clock() - self.start_game_clock  # 게임이 시작하고 나서 지난 시간
        self.delay = self.second_per_frame * self.cnt - self.game_time  # 기다려야 하는 시간(1초에 fps번만 캡쳐해야 함)
        
        if self.delay >= 0:
            sleep(self.delay)
        elif self.delay > -0.5:
            pass
        else:
            print("너무 느린 실행 속도")
            exit()
    
    def game_frame(self, sess, control_switch=True):
        self.t3 = clock()
        self.full_screenshot = _full_screenshot(self.windows, npsw=True)
        # screenshots.append(self.full_screenshot)
        check1 = np.all(self.full_screenshot[self.c_y1:self.c_y2, self.c_x1:self.c_x2] == self.check_game_over_img)
        check2 = np.all(self.full_screenshot[self.c_y1:self.c_y2, self.c_x1:self.c_x2] == self.check_lobby_img[0])
        check3 = np.all(self.full_screenshot[self.c_y1:self.c_y2, self.c_x1:self.c_x2] == self.check_lobby_img[1])
        if check1 or check2 or check3:
            self.end_game_time = datetime.strftime(datetime.today(), '%Y%m%d-%H%M%S')
            self.end_game_clock = clock()
            return False
        self.t4 = clock()
        
        self.t5 = clock()            
        self.current_clock = clock()  # 현재 시각을 저장함
        
        self.key_bool_li = []
        epsilon = (0.99 ** self.game_no)
        if(tnt.randf(0, 1) > epsilon) and self.ai_sw:# and False:
            img = self.full_screenshot[self.y1:self.y2 + 1, self.x1:self.x2 + 1,:3]
            # img = _pooling(4, 4, img, min_sw=True)
            img = _pooling(2, 2, img, min_sw=True)
            img = rgb2gray(img)
            # print(img, img.shape)
            img = img.flatten()
            # print(img, img.shape)
            # exit()
            
            q = sess.run(tnt.output_layer, feed_dict={tnt.X:[img]})
            # Find the max index (the chosen action).
            index = q.argmax()
            print(q, index)
            action = index  # + 1
            for i, v in enumerate(self.number2key_li[action]):
                if v and control_switch and self.key_state[i]!=v:
                    platformModule._keyDown(self.key_li2[i])
                elif self.key_state[i]!=v:
                    platformModule._keyUp(self.key_li2[i])
                self.key_bool_li.append(int(v))
                self.key_state[i] = v
        else:
            try:
                for i, v in enumerate(choice(self.number2key_li)):
                    if v and control_switch and self.key_state[i]!=v:
                        platformModule._keyDown(self.key_li2[i])
                    elif self.key_state[i]!=v:
                        platformModule._keyUp(self.key_li2[i])
                    self.key_bool_li.append(int(v))
                    self.key_state[i] = v
            except Exception as e:
                print('에러 발생 :', e)
                exit()
                
        self.t6 = clock()
        
        self.t1 = clock()
        try:
            self.score = self.check_score()
            self.level = self.check_level()
            self.line = self.check_line()
            self.next_piece = self.check_block()[0]
        except Exception as e:
            print('확인 실패 :', e)
            self.jiyeon()
            self.cnt += 1
            return True
        
        self.info_li.append({'current_clock':self.current_clock,
                         'score':self.score,
                         'level':self.level,
                         'line':self.line,
                         'key':self.key_bool_li,
                         # 'push_t':self.push_t_li,
                         'next_piece':self.next_piece,
                         'screenshot':self.full_screenshot})
        self.t2 = clock()
        print("%11.4f %5d %3d %4d %d %d %d %d %d %1s %7.5f %7.5f %7.5f %5d %7.6f" % (self.current_clock, self.score, self.level,
                                                                                     self.line, self.key_bool_li[0],
                                                                                     self.key_bool_li[1], self.key_bool_li[2],
                                                                                     self.key_bool_li[3], self.key_bool_li[4],
                                                                                     # self.push_t_li,
                                                                                     self.next_piece,
                                                                                     self.t2 - self.t1, self.t4 - self.t3,
                                                                                     self.t6 - self.t5, self.game_no, self.delay))
        # print(self.next_piece, t2 - t1)
        self.jiyeon()
        self.cnt += 1
        return True
    
    def game(self):
        print('창 위치 확인')
        self.windows = window_info('Not Tetris 2')
        print(self.windows)
        if self.windows == -1:
            print('실행 중인지 확인')
            exit()
            
        while True and False:
            t1 = clock()
            # full_screenshot = np.array(ImageGrab.grab(bbox=(windows.left, windows.top, windows.right, windows.bottom)))
            full_screenshot = _full_screenshot(self.windows, npsw=False)
            t2 = clock()
            print(t2 - t1)
            
        print('준비중...')
        c_x1, c_x2, c_y1, c_y2 = 28, 92, 46, 75
        self.c_x1, self.c_x2, self.c_y1, self.c_y2 = c_x1, c_x2, c_y1, c_y2
        self.check_lobby_img = [np.array(pilimg.open(r'img\check_lobby1.png'))[c_y1:c_y2, c_x1:c_x2],
                       np.array(pilimg.open(r'img\check_lobby2.png'))[c_y1:c_y2, c_x1:c_x2]]
        self.check_lobby = [False, False]
            
        print('로비인지 확인중')
        self.check_game_over_img = np.array(pilimg.open(r'img\check_game_over.png'))[c_y1:c_y2, c_x1:c_x2]
        while not all(self.check_lobby):
            self.full_screenshot = _full_screenshot(self.windows, npsw=True)
                
            for i, v in enumerate(self.check_lobby_img):
                if np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == v):
                    self.check_lobby[i] = True
        # print(check_lobby)
            
        print('게임 시작')
        _press('enter', 0)  # 게임 시작
        _press('enter', 0)  # 일시정지
        sleep(1)
        # print("%11s %5s %3s %4s %s %s %s %s %s %70s %1s %7s %7s %7s" % ('clock', 'score', 'lev', 'line',
                                                                              # 'z', 'x', 'l', 'r', 'd',
                                                                              # 'push_t', 'p', 'tForCal', 'tForCap', 'tForPus'))
        print("%11s %5s %3s %4s %s %s %s %s %s %1s %7s %7s %7s" % ('clock', 'score', 'lev', 'line',
                                                                              'z', 'x', 'l', 'r', 'd',
                                                                              'p', 'tForCal', 'tForCap', 'tForPus'))
        with tf.Session() as sess:
            try:
                self.saver.restore(sess, os.getcwd() + "/model.ckpt")
                self.ai_sw = True
            except:
                self.ai_sw = False
            key = None
            # self.key_li = []
            self.key_bool_li = []
            # push_t = None
            self.push_t_li = []
            self.current_clock = None
            # screenshots = []
            self.info_li = []
            # bool_li = [True, False]
            self.is_push = None
            self.cnt = 1  # 반복 횟수
            self.game_time = None
            self.fps = 10
            self.second_per_frame = 1 / self.fps  # 1초에 fps번 캡쳐 (및 조작)
            self.t1, self.t2 = None, None
            self.t3, self.t4 = None, None
            self.t5, self.t6 = None, None
            self.key_state = [False, False, False, False, False]
            self.delay = 0
        ############################################################################################################
        #=========================================================================================================
            _press('enter', 0)  # 일시정지 해제
            # self.full_screenshot = _full_screenshot(self.windows, npsw=True)
            self.start_game_time = datetime.strftime(datetime.today(), '%Y%m%d-%H%M%S')
            self.start_game_clock = clock()
            # print(self.start_game_clock)
            for i in range(3):  # DQN에 넣을 이미지를 캡쳐함
                self.game_frame(sess, control_switch=False)
                
            while self.game_frame(sess, control_switch=True):
                pass
        #=========================================================================================================
        ############################################################################################################
                
        print(np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_game_over_img),
              np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[0]),
              np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[1]))
        
        if np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_game_over_img):
            self.score = self.check_score()
            self.level = self.check_level()
            self.line = self.check_line()
            print(self.score, self.level, self.line)
        
        try:
            for i in self.key_li2:
                platformModule._keyUp(i)
        except Exception as e:
            print('에러 발생3 :', e)
            
        print('게임 종료')
        
        print('데이터 저장 중', clock())
        self.game_no = self.save_data()
        print('데이터 저장 완료', clock())
        
        print('학습 중', clock())
        tnt.main(self.game_no, self.game_no)  # 방금 진행한 판 학습
        print('학습 완료', clock())
            
        _press('left', 0.5)
        _press('down', 0.5)
        _press('left', 0.5)
        _press('down', 0.5)
        
        print('로비 대기', clock())
        while not np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[0]):  # 로비로 나왔는지 확인함
            self.full_screenshot = _full_screenshot(self.windows, npsw=True)
        
        # 신기록 달성인지 확인함
        self.check_lobby = [False, False]
        t1 = clock()
        while True:
            self.full_screenshot = _full_screenshot(self.windows, npsw=True)
            for i, v in enumerate(self.check_lobby_img):
                if np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == v):
                    self.check_lobby[i] = True
            if clock() - t1 > 3: break
                
        if not all(self.check_lobby):  # 신기록 달성시 이름 입력
            print('신기록 달성?', self.check_lobby)
            _press('down', 0)
            
            self.check_lobby = [False, False]
            t1 = clock()
            while True:
                self.full_screenshot = _full_screenshot(self.windows, npsw=True)
                for i, v in enumerate(self.check_lobby_img):
                    if np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == v):
                        self.check_lobby[i] = True
                if clock() - t1 > 3: break
            
            if not all(self.check_lobby):  # or True:
                print('신기록 달성', self.check_lobby)
                NTAI_NAME = datetime.strftime(datetime.today(), '%y%m%d')
                
                # now = datetime.strftime(datetime.today(), '%Y%m%d-%H%M%S') # 나중에 게임 종료 시각으로 수정하기
                createFolder(r'report\log')
                createFolder(r'report\new_record')
                self.full_screenshot = _full_screenshot(self.windows, npsw=False)
                self.full_screenshot.save(r'report\new_record\%s.png' % self.end_game_time)
                
                for i in NTAI_NAME:
                    _press(i, 0)
                _press('enter', 0)
                sleep(1)
                
                self.full_screenshot = _full_screenshot(self.windows, npsw=False)
                self.full_screenshot.save(r'report\new_record\%s_2.png' % self.end_game_time)


def rgb2gray(rgb):
    r, g, b = rgb[:,:, 0], rgb[:,:, 1], rgb[:,:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray  # .astype('uint8')


def createFolder(directory):  # 출처: https://data-make.tistory.com/170 [Data Makes Our Future]
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)
        

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    출처: https://everyday-deeplearning.tistory.com/entry/파이썬으로-딥러닝하기-CNNConvolution-Neural-Network [매일매일 딥러닝]
    다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    #"""
    N, C, H, W = input_data.shape
    # print(N, C, H, W)
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 위의 출력크기 공식을 이용하여 구현
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # print(N, C, filter_h, filter_w, out_h, out_w)
    # print(col)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:,:, y, x,:,:] = img[:,:, y:y_max:stride, x:x_max:stride]
            
    # print(col)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


class Pooling:  # 출처 : 밑바닥부터 시작하는 딥러닝(249p)

    def __init__(self, pool_h, pool_w, stride=1, pad=0, min_sw=True):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.min_sw = min_sw

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최소(대)값(2)
        if self.min_sw: out = np.min(col, axis=1)
        else: out = np.max(col, axis=1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


def _pooling(pool_x, pool_y, image, min_sw=True):  # 풀링 함수(풀링 필터의 가로 세로 크기와 이미지를 받아 풀링된 값을 반환함)
    temp = []
    temp.append(image[:,:, 0:3])
    image = np.array(temp)
    image = image.transpose(0, 3, 1, 2)
        
    __pooling = Pooling(pool_h=pool_y, pool_w=pool_x, stride=pool_x, min_sw=min_sw)
    image = __pooling.forward(image)
    image = image.transpose(0, 2, 3, 1)

    return image[0]


# sleep중에 남은 sleep 시간보다 짧은 입력 명령이 들어온다면 어떻게 해야? -> 그냥 알아서 해결될듯
# sleep중에 남은 sleep 시간보다 긴 입력 명령이 들어온다면 어떻게 해야? -> 중간에 끊김(그냥 놔둬도 괜찮을듯?)
# def th_press(key, s):
def _press(key, s):  # key를 s초 동안 눌렀다가 뗌
    while True:
        # pyautogui._failSafeCheck()
        try:
            # t1 = clock()
            # platformModule._keyUp(key) # 누르려는 키가 현재 눌려있는 경우를 대비
            platformModule._keyDown(key)
            # while True:
            sleep(s)
            platformModule._keyUp(key)
            break
        except Exception as e:
            print(e)
            print('_press() error')
            exit()
            pass


def window_info(target):
    titles = gw.getAllTitles()  # 현재 생성 되어있는 윈도우 창들의 타이틀 제목을 가져 온다.
    for i, name in enumerate(titles):
        # print(i, name)
        if name == target:
            return gw.getWindowsWithTitle(titles[0])[i]  # 목표 윈도우의 위치를 가져옴
    return -1


def _full_screenshot(windows, npsw=True):
    # npsw : numpy배열로 변환하여 반환하면 True
    x1 = windows.left  # +5
    y1 = windows.top  # +28
    x2 = windows.right
    y2 = windows.bottom
    if npsw: return np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
    else: return ImageGrab.grab(bbox=(x1, y1, x2, y2))


if __name__ == '__main__':
    oper = Operation()
    while True:
        oper.game()
