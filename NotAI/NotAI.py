from PIL import ImageGrab
import PIL.Image as pilimg
from time import perf_counter as clock, sleep
import pygetwindow as gw
import numpy as np
import os
from PIL import Image
from datetime import datetime
from random import choice
import _pyautogui_win as platformModule


class Operation:

    def __init__(self):
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
                self.block_data.append(_Pooling(self.POOLING_X, self.POOLING_Y,
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
        
        self.key_li = ['z', 'x', 'left', 'right', 'down', '']  # 조작키 리스트
        self.push_t = np.random.normal(0.5, 1, 1000)
        self.push_t = self.push_t[self.push_t > 0]  # 조작키를 누르는 시간(정규분포(0이하의 값들은 버림))
        
        self.windows = None
        self.full_screenshot = None
        self.check_lobby = None
        self.score, self.level, self.line = None, None, None
        self.next_piece = None
    
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
            avg = np.mean(self.full_screenshot[y1:y2, x1:x2], axis=2) # 스크린샷의 각 픽셀별 평균
            bo = avg == self.number_font # 숫자 이미지의 픽셀값과 정확하게 일치하는 픽셀(각각의 숫자 이미지에 대한 값을 모두 구함)
            bo = np.all(bo, axis=2) # 해당 줄의 픽셀값이 전부 같으면 True
            bo = np.all(bo, axis=1) # 해당 열의 픽셀값이 전부 같으면 True
            num = self.number_name[bo][0] # 최종적으로 픽셀값이 완전히 일치하는 숫자 이미지를 찾아냄
            score += 10 ** i * num # 점수 더하기
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
        pool_img = _Pooling(self.POOLING_X, self.POOLING_Y, self.full_screenshot[129:162, 122:155])
        
        cha = np.abs(self.block_data - pool_img)
        # print(cha.shape)
        hap = np.sum(np.sum(np.sum(cha, axis=3), axis=2), axis=1)
        bo = hap == np.min(hap)
        return self.block_data_label[bo]
    
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
        _press('enter', 0)
        _press('enter', 0)
        sleep(1)
        _press('enter', 0)
        # self.full_screenshot = _full_screenshot(self.windows, npsw=True)
        self.start_game_time = datetime.strftime(datetime.today(), '%Y%m%d-%H%M%S')
        self.start_game_clock = clock()
        ############################################################################################################
        #=========================================================================================================
        print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('현재 시각', '현재 점수', '현재 레벨', '현재 부순 줄', '방금 누른 키', '키를 누른 시간',
                                                            '다음 블록', '계산 시간'))
        key = None
        push_t = None
        current_clock = None
        # screenshots = []
        info_li = []
        while True:
            self.full_screenshot = _full_screenshot(self.windows, npsw=True)
            # screenshots.append(self.full_screenshot)
            if (np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_game_over_img) or np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[0]) or np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[1])):
                self.end_game_time = datetime.strftime(datetime.today(), '%Y%m%d-%H%M%S')
                self.end_game_clock = clock()
                break
            
            key = choice(self.key_li) # 누를키를 랜덤으로 선택함
            push_t = choice(self.push_t) # 특정 키를 누를 시간을 선택함
            current_clock = clock() # 현재 시각을 저장함
            _press(key, push_t) # 특정 키를 일정 시간동안 누름
            
            t1 = clock()
            self.score = self.check_score()
            self.level = self.check_level()
            self.line = self.check_line()
            self.next_piece = self.check_block()[0]
            
            info_li.append({'current_clock':current_clock,
                             'score':self.score,
                             'level':self.level,
                             'line':self.line,
                             'key':key,
                             'push_t':push_t,
                             'next_piece':self.next_piece,
                             'screenshot':self.full_screenshot})
            t2 = clock()
            print("%.4f\t%d\t%d\t%d\t%s\t%.6f\t%s\t%.6f" % (current_clock, self.score, self.level, self.line, key, push_t,
                                                            self.next_piece, t2-t1))
            # print(self.next_piece, t2 - t1)
        #=========================================================================================================
        ############################################################################################################
                
        print(np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_game_over_img), np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[0]), np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_lobby_img[1]))
        
        if np.all(self.full_screenshot[c_y1:c_y2, c_x1:c_x2] == self.check_game_over_img):
            self.score = self.check_score()
            self.level = self.check_level()
            self.line = self.check_line()
            print(self.score, self.level, self.line) 
            
        print('게임 종료')
        
        print('데이터 저장 중', clock())
        _dir = r'data\img\%s_%.6f' % (self.start_game_time, self.start_game_clock)
        createFolder(_dir)
        
        #          게임 시작 시간             _시작 클럭      조작 시작 시각        _조작 시간                
        # data\img\yyyymmdd-himiss_clock()\clock()(%.6f)_push_t(%.6f)_조작키.png
        # _dir = None
        _path = None
        # 추후 OracleDB에 바로 저장하도록 바꾸기(스크린샷은 그대로 폴더에 저장)
        createFolder(r'data\log')
        f = open(r'data\log\%s_%.6f.txt' % (self.start_game_time, self.start_game_clock), 'a', encoding='UTF-8')
        for i, v in enumerate(info_li):
            _path = _dir + r'\%.6f_%.6f_%s.png' % (v['current_clock'], v['push_t'], v['key'])
            Image.fromarray(v['screenshot'], 'RGB').save(_path)
            
            f.write('%.6f\t%.6f\t%s\t%d\t%d\t%d\t%s\n' % (v['current_clock'], v['push_t'], v['key']
                                                          , v['score'], v['level'], v['line'], v['next_piece']))
        f.close()
        print('데이터 저장 완료', clock())
            
        _press('left', 1)
            
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

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최소값(2)
        out = np.min(col, axis=1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


def _Pooling(pool_x, pool_y, image):  # 최소 풀링 함수(풀링 필터의 가로 세로 크기와 이미지를 받아 풀링된 값을 반환함)
    temp = []
    temp.append(image[:,:, 0:3])
    image = np.array(temp)
    image = image.transpose(0, 3, 1, 2)
        
    __pooling = Pooling(pool_h=pool_y, pool_w=pool_x, stride=pool_x)
    image = __pooling.forward(image)
    image = image.transpose(0, 2, 3, 1)

    return image[0]


def _press(key, s):  # key를 s초 동안 눌렀다가 뗌
    while True:
        # pyautogui._failSafeCheck()
        try:
            t1 = clock()
            platformModule._keyDown(key)
            # while True:
            sleep(s)
            platformModule._keyUp(key)
            break
        except:
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
