# -*- coding: utf-8 -*-
import numpy as np
from vt_pre import Ui_MainWindow
from PyQt5.QtCore import QDir,Qt
from PyQt5.QtWidgets import  QFileDialog, QMainWindow,QApplication,QMessageBox
from PyQt5.QtGui import QIntValidator
import pandas as pd
import cv2
from PyQt5.QtGui import QImage, QPixmap
import sys
import platform


s = platform.uname()
os_p = s[0]


def find_center(A,B,C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    return P,R

class query_window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.lineEdit_img_s2_2.setValidator(QIntValidator())

        self.ui.pushButton_selectsave_s2.clicked.connect(self.select_imageplace)
        self.ui.pushButton_openimg_s2.clicked.connect(self.openimg_s2)
        self.ui.background_label_s2.setCursor(Qt.CrossCursor)
        self.ui.background_label_s2.mousePressEvent = self.getPos
        self.ui.pushButton.clicked.connect(self.reset_point1)
        self.ui.pushButton_2.clicked.connect(self.reset_point2)
        self.ui.pushButton_3.clicked.connect(self.reset_point3)
        self.ui.listWidget.clicked.connect(self.set_num)
        self.ui.listWidget_2.clicked.connect(self.set_geno)
        self.ui.listWidget_3.clicked.connect(self.set_rep)
        self.ui.listWidget_4.clicked.connect(self.set_sex)
        self.ui.listWidget_5.clicked.connect(self.set_con)
        self.ui.pushButton_reset_s2_2.clicked.connect(self.nextp)
        self.ui.pushButton_reset_s2.clicked.connect(self.prevp)
        self.ui.pushButton_find_cyecle_s2.clicked.connect(self.find_cycle)
    def set_num(self):
        self.ui.lineEdit_img_s2_2.setText(self.ui.listWidget.currentItem().text())
    def set_geno(self):
        self.ui.lineEdit_img_s2_3.setText(self.ui.listWidget_2.currentItem().text())
    def set_rep(self):
        self.ui.lineEdit_img_s2_5.setText(self.ui.listWidget_3.currentItem().text())
    def set_sex(self):
        self.ui.lineEdit_img_s2_6.setText(self.ui.listWidget_4.currentItem().text())
    def set_con(self):
        self.ui.lineEdit_img_s2_4.setText(self.ui.listWidget_5.currentItem().text())
    def find_cycle(self):
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append([int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))])
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append([int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))])
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append([int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))])
        if len(xylist)<3:
            QMessageBox.information(self, 'Message', 'Please select three points at the edge of the cycle, exit', QMessageBox.Ok)
            return None
        if len(xylist)==3:
            red = (48, 48, 255)
            green = (34, 139, 34)
            yellow = (0, 255, 255)
            imageindex=self.ui.label_39.text().split(':')
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            imgname=df.loc[imageindex,'background']

            img=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
            P,R=find_center(xylist[0],xylist[1],xylist[2])
            x,y=P
            x=int(x)
            y=int(y)
            R=int(R)
            cv2.circle(img, (int(x),int(y)), int(R), (255, 255, 255), 2)
            cv2.circle(img, (int(x),int(y)), 3, red, -1, cv2.LINE_AA)

            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))


    def nextp(self):
        suc=True
        if self.ui.label_xdim_s2.text()=='None':
            suc=False
        if self.ui.label_xdim_p2_s2.text()=='None':
            suc=False
        if self.ui.label_xdim_p3_s2.text()=='None':
            suc=False
        if suc:
            imageindex=self.ui.label_39.text().split(':')
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            df.loc[imageindex,'x1']=int(float(self.ui.label_xdim_s2.text()))
            df.loc[imageindex,'y1']=int(float(self.ui.label_ydim_s2.text()))
            df.loc[imageindex,'x2']=int(float(self.ui.label_xdim_p2_s2.text()))
            df.loc[imageindex,'y2']=int(float(self.ui.label_ydim_p2_s2.text()))
            df.loc[imageindex,'x3']=int(float(self.ui.label_xdim_p3_s2.text()))
            df.loc[imageindex,'y3']=int(float(self.ui.label_ydim_p3_s2.text()))
            if self.ui.lineEdit_img_s2_2.text()!='':
                df.loc[imageindex,'num']=int(float(self.ui.lineEdit_img_s2_2.text()))
            if self.ui.lineEdit_img_s2_3.text()!='':
                df.loc[imageindex,'genotype']=self.ui.lineEdit_img_s2_3.text()
            if self.ui.lineEdit_img_s2_5.text()!='':
                df.loc[imageindex,'replicate']=self.ui.lineEdit_img_s2_5.text()
            if self.ui.lineEdit_img_s2_6.text()!='':
                df.loc[imageindex,'sex']=self.ui.lineEdit_img_s2_6.text()
            if self.ui.lineEdit_img_s2_4.text()!='':
                df.loc[imageindex,'condition']=self.ui.lineEdit_img_s2_4.text()
            if self.ui.checkBox.isChecked():
                df.loc[imageindex,'drop']=1
            else:
                df.loc[imageindex,'drop']=0
            df.to_csv(tmpdatafilename,index=False)

            imageindex=imageindex+1
            if imageindex<len(df):
                self.ui.checkBox.setChecked(False)
                self.ui.label_xdim_s2.setText('None')
                self.ui.label_xdim_p2_s2.setText('None')
                self.ui.label_xdim_p3_s2.setText('None')
                self.ui.label_ydim_s2.setText('None')
                self.ui.label_ydim_p2_s2.setText('None')
                self.ui.label_ydim_p3_s2.setText('None')

                imgname=df.loc[imageindex,'background']
                frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
                self.ui.label_37.setText('Image info:'+imgname)
                self.ui.label_39.setText('Image index:'+str(imageindex))
                if df.loc[imageindex,'drop']==1:
                    self.ui.checkBox.setChecked(True)
                num_list=list(set(list(df.loc[~df['num'].isna(),'num'])))
                if len(num_list)>0:
                    new_num_list=[str(x) for x in num_list]
                    new_num_list=list(set(new_num_list))
                    self.ui.listWidget.clear()
                    self.ui.listWidget.addItems(new_num_list)
                genotype_list=list(set(list(df.loc[~df['genotype'].isna(),'genotype'])))
                if len(genotype_list)>0:
                    new_genotype_list=[str(x) for x in genotype_list]
                    new_genotype_list=list(set(new_genotype_list))
                    self.ui.listWidget_2.clear()
                    self.ui.listWidget_2.addItems(new_genotype_list)

                rep_list=list(set(list(df.loc[~df['replicate'].isna(),'replicate'])))
                if len(rep_list)>0:
                    new_rep_list=[str(x) for x in rep_list]
                    new_rep_list=list(set(new_rep_list))
                    self.ui.listWidget_3.clear()
                    self.ui.listWidget_3.addItems(new_rep_list)
                sex_list=list(set(list(df.loc[~df['sex'].isna(),'sex'])))
                if len(sex_list)>0:
                    new_sex_list=[str(x) for x in sex_list]
                    new_sex_list=list(set(new_sex_list))
                    self.ui.listWidget_4.clear()
                    self.ui.listWidget_4.addItems(new_sex_list)
                condition_list=list(set(list(df.loc[~df['condition'].isna(),'condition'])))
                if len(condition_list)>0:
                    new_condition_list=[str(x) for x in condition_list]
                    new_condition_list=list(set(new_condition_list))
                    self.ui.listWidget_5.clear()
                    self.ui.listWidget_5.addItems(new_condition_list)
                if str(df.loc[imageindex,'num'])!='' and str(df.loc[imageindex,'num'])!='None' and str(df.loc[imageindex,'num'])!='nan':
                    self.ui.lineEdit_img_s2_2.setText(str(df.loc[imageindex,'num']))
                if str(df.loc[imageindex,'genotype'])!='' and str(df.loc[imageindex,'genotype'])!='None' and str(df.loc[imageindex,'genotype'])!='nan':
                    self.ui.lineEdit_img_s2_3.setText(str(df.loc[imageindex,'genotype']))
                if str(df.loc[imageindex,'replicate'])!='' and str(df.loc[imageindex,'replicate'])!='None' and str(df.loc[imageindex,'replicate'])!='nan':
                    self.ui.lineEdit_img_s2_5.setText(str(df.loc[imageindex,'replicate']))
                if str(df.loc[imageindex,'sex'])!='' and str(df.loc[imageindex,'sex'])!='None' and str(df.loc[imageindex,'sex'])!='nan':
                    self.ui.lineEdit_img_s2_6.setText(str(df.loc[imageindex,'sex']))
                if str(df.loc[imageindex,'condition'])!='' and str(df.loc[imageindex,'condition'])!='None' and str(df.loc[imageindex,'condition'])!='nan':
                    self.ui.lineEdit_img_s2_4.setText(str(df.loc[imageindex,'condition']))

                if str(df.loc[imageindex,'x1'])!='' and str(df.loc[imageindex,'x1'])!='None' and str(df.loc[imageindex,'x1'])!='nan':
                    self.ui.label_xdim_s2.setText(str(df.loc[imageindex,'x1']))
                if str(df.loc[imageindex,'x2'])!='' and str(df.loc[imageindex,'x2'])!='None' and str(df.loc[imageindex,'x2'])!='nan':
                    self.ui.label_xdim_p2_s2.setText(str(df.loc[imageindex,'x2']))
                if str(df.loc[imageindex,'x3'])!='' and str(df.loc[imageindex,'x3'])!='None' and str(df.loc[imageindex,'x3'])!='nan':
                    self.ui.label_xdim_p3_s2.setText(str(df.loc[imageindex,'x3']))
                if str(df.loc[imageindex,'y1'])!='' and str(df.loc[imageindex,'y1'])!='None' and str(df.loc[imageindex,'y1'])!='nan':
                    self.ui.label_ydim_s2.setText(str(df.loc[imageindex,'y1']))
                if str(df.loc[imageindex,'y2'])!='' and str(df.loc[imageindex,'y2'])!='None' and str(df.loc[imageindex,'y2'])!='nan':
                    self.ui.label_ydim_p2_s2.setText(str(df.loc[imageindex,'y2']))
                if str(df.loc[imageindex,'y3'])!='' and str(df.loc[imageindex,'y3'])!='None' and str(df.loc[imageindex,'y3'])!='nan':
                    self.ui.label_ydim_p3_s2.setText(str(df.loc[imageindex,'y3']))

            else:
                QMessageBox.information(self, 'Message', 'Job finished', QMessageBox.Ok)
        else:
            QMessageBox.information(self, 'Message', 'Please mark the cycle in the background image', QMessageBox.Ok)

    def prevp(self):
        suc=True
        if self.ui.label_xdim_s2.text()=='None':
            suc=False
        if self.ui.label_xdim_p2_s2.text()=='None':
            suc=False
        if self.ui.label_xdim_p3_s2.text()=='None':
            suc=False
        if suc:
            imageindex=self.ui.label_39.text().split(':')
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            df.loc[imageindex,'x1']=int(float(self.ui.label_xdim_s2.text()))
            df.loc[imageindex,'y1']=int(float(self.ui.label_ydim_s2.text()))
            df.loc[imageindex,'x2']=int(float(self.ui.label_xdim_p2_s2.text()))
            df.loc[imageindex,'y2']=int(float(self.ui.label_ydim_p2_s2.text()))
            df.loc[imageindex,'x3']=int(float(self.ui.label_xdim_p3_s2.text()))
            df.loc[imageindex,'y3']=int(float(self.ui.label_ydim_p3_s2.text()))
            if self.ui.lineEdit_img_s2_2.text()!='':
                df.loc[imageindex,'num']=int(float(self.ui.lineEdit_img_s2_2.text()))
            if self.ui.lineEdit_img_s2_3.text()!='':
                df.loc[imageindex,'genotype']=self.ui.lineEdit_img_s2_3.text()
            if self.ui.lineEdit_img_s2_5.text()!='':
                df.loc[imageindex,'replicate']=self.ui.lineEdit_img_s2_5.text()
            if self.ui.lineEdit_img_s2_6.text()!='':
                df.loc[imageindex,'sex']=self.ui.lineEdit_img_s2_6.text()
            if self.ui.lineEdit_img_s2_4.text()!='':
                df.loc[imageindex,'condition']=self.ui.lineEdit_img_s2_4.text()
            if self.ui.checkBox.isChecked():
                df.loc[imageindex,'drop']=1
            else:
                df.loc[imageindex,'drop']=0
            df.to_csv(tmpdatafilename,index=False)

            imageindex=imageindex-1
            if imageindex>=0:
                self.ui.checkBox.setChecked(False)
                self.ui.label_xdim_s2.setText('None')
                self.ui.label_xdim_p2_s2.setText('None')
                self.ui.label_xdim_p3_s2.setText('None')
                self.ui.label_ydim_s2.setText('None')
                self.ui.label_ydim_p2_s2.setText('None')
                self.ui.label_ydim_p3_s2.setText('None')

                imgname=df.loc[imageindex,'background']

                frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
                self.ui.label_37.setText('Image info:'+imgname)
                self.ui.label_39.setText('Image index:'+str(imageindex))

                if df.loc[imageindex,'drop']==1:
                    self.ui.checkBox.setChecked(True)
                num_list=list(set(list(df.loc[~df['num'].isna(),'num'])))
                if len(num_list)>0:
                    new_num_list=[str(x) for x in num_list]
                    new_num_list=list(set(new_num_list))
                    self.ui.listWidget.clear()
                    self.ui.listWidget.addItems(new_num_list)
                genotype_list=list(set(list(df.loc[~df['genotype'].isna(),'genotype'])))
                if len(genotype_list)>0:
                    new_genotype_list=[str(x) for x in genotype_list]
                    new_genotype_list=list(set(new_genotype_list))
                    self.ui.listWidget_2.clear()
                    self.ui.listWidget_2.addItems(new_genotype_list)

                rep_list=list(set(list(df.loc[~df['replicate'].isna(),'replicate'])))
                if len(rep_list)>0:
                    new_rep_list=[str(x) for x in rep_list]
                    new_rep_list=list(set(new_rep_list))
                    self.ui.listWidget_3.clear()
                    self.ui.listWidget_3.addItems(new_rep_list)
                sex_list=list(set(list(df.loc[~df['sex'].isna(),'sex'])))
                if len(sex_list)>0:
                    new_sex_list=[str(x) for x in sex_list]
                    new_sex_list=list(set(new_sex_list))
                    self.ui.listWidget_4.clear()
                    self.ui.listWidget_4.addItems(new_sex_list)
                condition_list=list(set(list(df.loc[~df['condition'].isna(),'condition'])))
                if len(condition_list)>0:
                    new_condition_list=[str(x) for x in condition_list]
                    new_condition_list=list(set(new_condition_list))
                    self.ui.listWidget_5.clear()
                    self.ui.listWidget_5.addItems(new_condition_list)
                if str(df.loc[imageindex,'num'])!='' and str(df.loc[imageindex,'num'])!='None' and str(df.loc[imageindex,'num'])!='nan':
                    self.ui.lineEdit_img_s2_2.setText(str(df.loc[imageindex,'num']))
                if str(df.loc[imageindex,'genotype'])!='' and str(df.loc[imageindex,'genotype'])!='None' and str(df.loc[imageindex,'genotype'])!='nan':
                    self.ui.lineEdit_img_s2_3.setText(str(df.loc[imageindex,'genotype']))
                if str(df.loc[imageindex,'replicate'])!='' and str(df.loc[imageindex,'replicate'])!='None' and str(df.loc[imageindex,'replicate'])!='nan':
                    self.ui.lineEdit_img_s2_5.setText(str(df.loc[imageindex,'replicate']))
                if str(df.loc[imageindex,'sex'])!='' and str(df.loc[imageindex,'sex'])!='None' and str(df.loc[imageindex,'sex'])!='nan':
                    self.ui.lineEdit_img_s2_6.setText(str(df.loc[imageindex,'sex']))
                if str(df.loc[imageindex,'condition'])!='' and str(df.loc[imageindex,'condition'])!='None' and str(df.loc[imageindex,'condition'])!='nan':
                    self.ui.lineEdit_img_s2_4.setText(str(df.loc[imageindex,'condition']))

                if str(df.loc[imageindex,'x1'])!='' and str(df.loc[imageindex,'x1'])!='None' and str(df.loc[imageindex,'x1'])!='nan':
                    self.ui.label_xdim_s2.setText(str(df.loc[imageindex,'x1']))
                if str(df.loc[imageindex,'x2'])!='' and str(df.loc[imageindex,'x2'])!='None' and str(df.loc[imageindex,'x2'])!='nan':
                    self.ui.label_xdim_p2_s2.setText(str(df.loc[imageindex,'x2']))
                if str(df.loc[imageindex,'x3'])!='' and str(df.loc[imageindex,'x3'])!='None' and str(df.loc[imageindex,'x3'])!='nan':
                    self.ui.label_xdim_p3_s2.setText(str(df.loc[imageindex,'x3']))
                if str(df.loc[imageindex,'y1'])!='' and str(df.loc[imageindex,'y1'])!='None' and str(df.loc[imageindex,'y1'])!='nan':
                    self.ui.label_ydim_s2.setText(str(df.loc[imageindex,'y1']))
                if str(df.loc[imageindex,'y2'])!='' and str(df.loc[imageindex,'y2'])!='None' and str(df.loc[imageindex,'y2'])!='nan':
                    self.ui.label_ydim_p2_s2.setText(str(df.loc[imageindex,'y2']))
                if str(df.loc[imageindex,'y3'])!='' and str(df.loc[imageindex,'y3'])!='None' and str(df.loc[imageindex,'y3'])!='nan':
                    self.ui.label_ydim_p3_s2.setText(str(df.loc[imageindex,'y3']))


            else:
                QMessageBox.information(self, 'Message', 'First image', QMessageBox.Ok)
        else:
            QMessageBox.information(self, 'Message', 'Please mark the cycle in the background image', QMessageBox.Ok)


    def select_imageplace(self):
        directory1 = QFileDialog.getExistingDirectory(self, "select folder", "/")
        print(directory1)
        if os_p == 'Windows':
            tmpdatafilename = directory1.replace('/', '\\')
        else:
            tmpdatafilename = directory1
        self.ui.lineEdit_select_save_s2.setText(tmpdatafilename)

    def openimg_s2(self):
        if self.ui.lineEdit_select_save_s2.text()!='':
            dig = QFileDialog()
            dig.setNameFilters(["metadata file(*.csv)"])
            dig.setFileMode(QFileDialog.ExistingFile)
            dig.setFilter(QDir.Files)
            if dig.exec_():
                filenames = dig.selectedFiles()
                if os_p == 'Windows':
                    tmpdatafilename = filenames[0].replace('/', '\\')
                else:
                    tmpdatafilename = filenames[0]
                self.ui.lineEdit_img_s2.setText(tmpdatafilename)
                df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
                imgname=df.loc[0,'background']
                frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
                self.ui.label_37.setText('Image info:'+imgname)
                self.ui.label_39.setText('Image index:'+str(0))
                if df.loc[0,'drop']==1:
                    self.ui.checkBox.setChecked(True)
                num_list=list(set(list(df.loc[~df['num'].isna(),'num'])))
                if len(num_list)>0:
                    new_num_list=[str(x) for x in num_list]
                    new_num_list=list(set(new_num_list))
                    self.ui.listWidget.clear()
                    self.ui.listWidget.addItems(new_num_list)
                genotype_list=list(set(list(df.loc[~df['genotype'].isna(),'genotype'])))
                if len(genotype_list)>0:
                    new_genotype_list=[str(x) for x in genotype_list]
                    new_genotype_list=list(set(new_genotype_list))
                    self.ui.listWidget_2.clear()
                    self.ui.listWidget_2.addItems(new_genotype_list)

                rep_list=list(set(list(df.loc[~df['replicate'].isna(),'replicate'])))
                if len(rep_list)>0:
                    new_rep_list=[str(x) for x in rep_list]
                    new_rep_list=list(set(new_rep_list))
                    self.ui.listWidget_3.clear()
                    self.ui.listWidget_3.addItems(new_rep_list)
                sex_list=list(set(list(df.loc[~df['sex'].isna(),'sex'])))
                if len(sex_list)>0:
                    new_sex_list=[str(x) for x in sex_list]
                    new_sex_list=list(set(new_sex_list))
                    self.ui.listWidget_4.clear()
                    self.ui.listWidget_4.addItems(new_sex_list)
                condition_list=list(set(list(df.loc[~df['condition'].isna(),'condition'])))
                if len(condition_list)>0:
                    new_condition_list=[str(x) for x in condition_list]
                    new_condition_list=list(set(new_condition_list))
                    self.ui.listWidget_5.clear()
                    self.ui.listWidget_5.addItems(new_condition_list)
                if str(df.loc[0,'num'])!='' and str(df.loc[0,'num'])!='None' and str(df.loc[0,'num'])!='nan':
                    self.ui.lineEdit_img_s2_2.setText(str(df.loc[0,'num']))
                if str(df.loc[0,'genotype'])!='' and str(df.loc[0,'genotype'])!='None' and str(df.loc[0,'genotype'])!='nan':
                    self.ui.lineEdit_img_s2_3.setText(str(df.loc[0,'genotype']))
                if str(df.loc[0,'replicate'])!='' and str(df.loc[0,'replicate'])!='None' and str(df.loc[0,'replicate'])!='nan':
                    self.ui.lineEdit_img_s2_5.setText(str(df.loc[0,'replicate']))
                if str(df.loc[0,'sex'])!='' and str(df.loc[0,'sex'])!='None' and str(df.loc[0,'sex'])!='nan':
                    self.ui.lineEdit_img_s2_6.setText(str(df.loc[0,'sex']))
                if str(df.loc[0,'condition'])!='' and str(df.loc[0,'condition'])!='None' and str(df.loc[0,'condition'])!='nan':
                    self.ui.lineEdit_img_s2_4.setText(str(df.loc[0,'condition']))

                if str(df.loc[0,'x1'])!='' and str(df.loc[0,'x1'])!='None' and str(df.loc[0,'x1'])!='nan':
                    self.ui.label_xdim_s2.setText(str(df.loc[0,'x1']))
                if str(df.loc[0,'x2'])!='' and str(df.loc[0,'x2'])!='None' and str(df.loc[0,'x2'])!='nan':
                    self.ui.label_xdim_p2_s2.setText(str(df.loc[0,'x2']))
                if str(df.loc[0,'x3'])!='' and str(df.loc[0,'x3'])!='None' and str(df.loc[0,'x3'])!='nan':
                    self.ui.label_xdim_p3_s2.setText(str(df.loc[0,'x3']))
                if str(df.loc[0,'y1'])!='' and str(df.loc[0,'y1'])!='None' and str(df.loc[0,'y1'])!='nan':
                    self.ui.label_ydim_s2.setText(str(df.loc[0,'y1']))
                if str(df.loc[0,'y2'])!='' and str(df.loc[0,'y2'])!='None' and str(df.loc[0,'y2'])!='nan':
                    self.ui.label_ydim_p2_s2.setText(str(df.loc[0,'y2']))
                if str(df.loc[0,'y3'])!='' and str(df.loc[0,'y3'])!='None' and str(df.loc[0,'y3'])!='nan':
                    self.ui.label_ydim_p3_s2.setText(str(df.loc[0,'y3']))
        else:
            QMessageBox.information(self, 'Message', 'Please set the image storage place', QMessageBox.Ok)

    def getPos(self , event):
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))))
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))))
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))))

        red = (48, 48, 255)
        green = (34, 139, 34)
        yellow = (0, 255, 255)
        color=[red,green,yellow]
        font = cv2.FONT_HERSHEY_SIMPLEX
        x = max(int(event.pos().x()),1)
        y = max(int(event.pos().y())-6,1)
        # print(x,y)
        imageindex=self.ui.label_39.text().split(':')
        if len(xylist)<3 and len(imageindex[1])>0:
            xylist.append((x,y))
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            imgname=df.loc[imageindex,'background']
            frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
            for i in range(len(xylist)):
                cv2.circle(frame, xylist[i], 3, color[i], -1, cv2.LINE_AA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
            self.ui.label_37.setText('Image info:'+imgname)
            self.ui.label_39.setText('Image index:'+str(imageindex))

            fill=False
            if (not fill) and self.ui.label_xdim_s2.text()=='None':
                self.ui.label_xdim_s2.setText(str(x))
                self.ui.label_ydim_s2.setText(str(y))
                fill=True
            if (not fill) and self.ui.label_xdim_p2_s2.text()=='None':
                self.ui.label_xdim_p2_s2.setText(str(x))
                self.ui.label_ydim_p2_s2.setText(str(y))
                fill=True
            if (not fill) and self.ui.label_xdim_p3_s2.text()=='None':
                self.ui.label_xdim_p3_s2.setText(str(x))
                self.ui.label_ydim_p3_s2.setText(str(y))
                fill=True
    def reset_point1(self , event):
        self.ui.label_xdim_s2.setText('None')
        self.ui.label_ydim_s2.setText('None')
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))))
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))))
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))))

        red = (48, 48, 255)
        green = (34, 139, 34)
        yellow = (0, 255, 255)
        color=[red,green,yellow]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print(x,y)
        if len(xylist)<3:
            imageindex=self.ui.label_39.text().split(':')
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            imgname=df.loc[imageindex,'background']
            frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
            for i in range(len(xylist)):
                cv2.circle(frame, xylist[i], 3, color[i], -1, cv2.LINE_AA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
            self.ui.label_37.setText('Image info:'+imgname)
            self.ui.label_39.setText('Image index:'+str(imageindex))
    def reset_point2(self , event):
        self.ui.label_xdim_p2_s2.setText('None')
        self.ui.label_ydim_p2_s2.setText('None')
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))))
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))))
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))))

        red = (48, 48, 255)
        green = (34, 139, 34)
        yellow = (0, 255, 255)
        color=[red,green,yellow]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print(x,y)
        if len(xylist)<3:
            imageindex=self.ui.label_39.text().split(':')
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            imgname=df.loc[imageindex,'background']
            frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
            for i in range(len(xylist)):
                cv2.circle(frame, xylist[i], 3, color[i], -1, cv2.LINE_AA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
            self.ui.label_37.setText('Image info:'+imgname)
            self.ui.label_39.setText('Image index:'+str(imageindex))

    def reset_point3(self , event):
        self.ui.label_xdim_p3_s2.setText('None')
        self.ui.label_ydim_p3_s2.setText('None')
        xylist=[]
        if self.ui.label_xdim_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_s2.text())),int(float(self.ui.label_ydim_s2.text()))))
        if self.ui.label_xdim_p2_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p2_s2.text())),int(float(self.ui.label_ydim_p2_s2.text()))))
        if self.ui.label_xdim_p3_s2.text()!='None':
            xylist.append((int(float(self.ui.label_xdim_p3_s2.text())),int(float(self.ui.label_ydim_p3_s2.text()))))

        red = (48, 48, 255)
        green = (34, 139, 34)
        yellow = (0, 255, 255)
        color=[red,green,yellow]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print(x,y)
        if len(xylist)<3:
            imageindex=self.ui.label_39.text().split(':')
            imageindex=int(float(imageindex[1]))
            tmpdatafilename=self.ui.lineEdit_img_s2.text()
            df=pd.read_csv(tmpdatafilename,header=0,index_col=None)
            imgname=df.loc[imageindex,'background']
            frame=cv2.imread(self.ui.lineEdit_select_save_s2.text()+'/'+imgname,cv2.IMREAD_COLOR)
            for i in range(len(xylist)):
                cv2.circle(frame, xylist[i], 3, color[i], -1, cv2.LINE_AA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.background_label_s2.setPixmap(QPixmap.fromImage(img))
            self.ui.label_37.setText('Image info:'+imgname)
            self.ui.label_39.setText('Image index:'+str(imageindex))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = query_window()
    window.show()
    sys.exit(app.exec_())
