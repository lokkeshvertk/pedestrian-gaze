from xml.dom.minidom import ReadOnlySequentialNamedNodeMap
from cv2 import line
from string import *
from matplotlib.pyplot import prism
import numpy as np 
import xlwt
from xlwt import Workbook
from pathlib import Path


participant_number = 0
for pnum in range(9):
    if pnum == 0: continue
    for path in Path('/home/tklokkeshver/Test_data/Participant'+str(pnum)).rglob('*.txt'):
        file_name = str(path)
        new_name = file_name.strip().split('/')[5].strip().split('.')[0]


        wb = Workbook()
        sheet1 = wb.add_sheet('Trial 1')
        sheet1.write(0,0, 'Video Frame')
        sheet1.write(0,1, 'Vehicle')
        sheet1.write(0,2, 'Confidence')
        sheet1.write(0,3, 'LX')
        sheet1.write(0,4, 'TY')
        sheet1.write(0,5, 'Width')
        sheet1.write(0,6, 'Height')
        sheet1.write(0,7, 'Gaze in x')
        sheet1.write(0,8, 'Gaze in y')
        sheet1.write(0,9, 'Vehicle Status')
        sheet1.write(0,10, 'Pixel Difference')
        sheet1.write(0,11, 'Attention Status')
        sheet1.write(0,12, 'Attention Value')
    
        # file_name = ("/home/tklokkeshver/pilot11.txt") #str("trial" + "{0:06}".format(i)+".txt")
        #file_name = ("/home/tklokkeshver/PILOT/1.txt")
        participant_number += 1
        new_line = []
        row_num = 0
        prev_text = " "
        vehicle_type = []
        confi = []
        x = []
        y = []
        w = []
        h = []
        gx = 0
        gy = 0
        fwd = ""
        swd = ""
        value = 0
        a_status = ""
        a_value = 0
        v_status = ""
        v_value = 0
        frame_number = 0

        with open (file_name) as files:
            for line in files:
                new_line.extend(line.strip().split('------------------------------------------------------------------------'))
            for element in new_line:
                if element == 'Objects:':
                    row_num += 1
                    frame_number += 1
                    if len(vehicle_type) <= 1:
                        if len(vehicle_type) == 0:
                            sheet1.write(row_num,0,frame_number)
                            sheet1.write(row_num,1, '-')
                            sheet1.write(row_num,2, '0')
                            sheet1.write(row_num,3, '0')
                            sheet1.write(row_num,4, '0')
                            sheet1.write(row_num,5, '0')
                            sheet1.write(row_num,6, '0')
                            sheet1.write(row_num,7,gx)
                            sheet1.write(row_num,8,gy)
                            sheet1.write(row_num,9,v_status)
                            sheet1.write(row_num,10,v_value)
                            sheet1.write(row_num,11,a_status)
                            sheet1.write(row_num,12,a_value)
                            continue
                        else:
                            sheet1.write(row_num,0,frame_number)
                            sheet1.write(row_num,1, vehicle_type[0])
                            sheet1.write(row_num,2, confi[0])
                            sheet1.write(row_num,3, x[0])
                            sheet1.write(row_num,4, y[0])
                            sheet1.write(row_num,5, w[0])
                            sheet1.write(row_num,6, h[0])
                            sheet1.write(row_num,7,gx)
                            sheet1.write(row_num,8,gy)
                            sheet1.write(row_num,9,v_status)
                            sheet1.write(row_num,10,v_value)
                            sheet1.write(row_num,11,a_status)
                            sheet1.write(row_num,12,a_value)
                    else:
                        for i in range(len(vehicle_type)):
                            sheet1.write(row_num,0,frame_number)
                            sheet1.write(row_num,1, vehicle_type[i])
                            sheet1.write(row_num,2, confi[i])
                            sheet1.write(row_num,3, x[i])
                            sheet1.write(row_num,4, y[i])
                            sheet1.write(row_num,5, w[i])
                            sheet1.write(row_num,6, h[i])
                            row_num += 1
                        row_num -=1
                        sheet1.write(row_num,7,gx)
                        sheet1.write(row_num,8,gy)
                        sheet1.write(row_num,9,v_status)
                        sheet1.write(row_num,10,v_value)
                        sheet1.write(row_num,11,a_status)
                        sheet1.write(row_num,12,a_value)
                    vehicle_type = []
                    confi = []
                    x = []
                    y = []
                    w = []
                    h = []
                    continue
                elif len(element) != 0 and element != 'Objects:':          
                    texts = element.split()
                    first_word = texts[0]
                    if first_word == 'Gaze:':
                        gv = texts[1:3]
                        gx = ((gv[0].strip().split('[')[1]).strip().split(',')[0])
                        gy = (gv[1].strip().split(']')[0])
                    if first_word == 'Vehicle':
                        vsts = texts[2:4]
                        vsts_value = texts[5:6]                        
                        if len(texts) == 4:
                            fwd = (vsts[0].strip().split('[')[1]).strip().split('\'')[1]
                            swd = (vsts[1].strip().split(']')[0]).strip().split('\'')[0]
                            value = 0
                        if len(texts) == 6:
                            fwd = (vsts[0].strip().split('[')[1]).strip().split('\'')[1]
                            swd = vsts[1]
                            value = (vsts_value[0].strip().split(']')[0]).strip().split('\'')[0]
                        v_status = fwd+ " "+ swd
                        v_value = value
                        #print(v_status, v_value)
                    if first_word == 'Attention':
                        asts = texts[2:5]
                        if len(texts) == 4:
                            first = ((asts[0].strip().split('[')[1]).strip().split(',')[0]).strip().split('\'')[1]
                            second = (asts[1].strip().split(']'))[0]
                            a_status = first
                            a_value = second
                        else:
                            first = (asts[0].strip().split('[')[1]).strip().split('\'')[1]
                            second = ((asts[1].strip().split(',')[0]).strip().split('\'')[0]).strip().split(']')[0]
                            third = asts[2].strip().split(']')[0]
                            a_status = first+" "+second
                            a_value = third
                        #print(a_status, a_value)
                    if first_word != 'Attention' and first_word != 'Vehicle' and first_word != 'Gaze:':
                        #if first_word == 'Car' or first_word == 'Van'or first_word == 'Truck' or  first_word == 'Cyclist' or first_word == 'Pedestrian'or first_word == 'Person_sitting'or first_word == 'Tram' or first_word == 'Misc':
                        if len(texts) == 10:
                            vt = str(texts[0]).strip().split(':')
                            lv = str(texts[9]).strip().split(')')
                            vehicle_type.append(vt[0])
                            confi.append(texts[1])
                            x.append(texts[3])
                            y.append(texts[5])
                            w.append(texts[7])
                            h.append(lv[0])
                else: continue


        wb.save('/home/tklokkeshver/Test_Datasheet/P'+ str(pnum) +'/'+ new_name + '.xls')