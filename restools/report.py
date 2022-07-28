from datetime import *
import calendar
import sys
import os
import os.path
import subprocess
import json

import matplotlib.pyplot as plt


class QuickReport:
    """
    Class for making quick reports
    """

    TEX_FILE_HEADER = r'''\documentclass[11pt]{article}
\usepackage{amsfonts,longtable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{esint}
\usepackage{mathdesign}
\usepackage{graphicx}
\usepackage{float}
\usepackage{bm}
\usepackage{hyperref}
\usepackage{epstopdf}
\usepackage[export]{adjustbox}
\textwidth=16cm
\oddsidemargin=1cm
\textheight=21cm
\topmargin=-1cm
\setlength\parindent{0pt}
\setcounter{MaxMatrixCols}{20}
'''
    TEX_FILE_FOOTER = r'\end{document}'
    REPORTS_ROOT = os.path.join('reports')
    IMAGES_DIR = 'images'

    def __init__(self, title):
        self.title = title
        self.blocks = []
        if not os.path.exists(QuickReport.REPORTS_ROOT):
            os.mkdir(QuickReport.REPORTS_ROOT)
        report_path = self._get_report_path()
#        if os.path.exists(report_path):
#            shutil.rmtree(report_path)
        os.mkdir(report_path)

    def add_section(self, section_name):
        self.blocks.append(r'\section{' + section_name + '}\n')

    def add_comment(self, comment):
        self.blocks.append(comment)

    def add_plot(self, fig, axes, caption, comment = ''):
        images_path = self._get_images_path()
        if not os.path.exists(images_path):
            os.mkdir(images_path)

        block_number = len(self.blocks)
        block_pic_filename = str(block_number) + '.png'
        block_pic_path = os.path.join(images_path, block_pic_filename)
        block_pic_latex_path = QuickReport.IMAGES_DIR + '/' + block_pic_filename
        figure_label = 'fig:' + str(block_number)
        fig.tight_layout()
        fig.savefig(block_pic_path)
        plt.clf()
        block_content = comment.replace('\\ref{@this@}', '\\ref{' + figure_label + '}') + '\n'
        block_content += '\\begin{figure}[H]\n'
        block_content += r'\includegraphics[width=' + str(fig.get_figwidth()/7.) + r'\textwidth, center]{' + block_pic_latex_path + '}\n'
        block_content += r'\caption{' + caption + r'.}\label{' + figure_label + '}\n\end{figure}\n'

        self.blocks.append(block_content)

    def print_out(self):
        if not os.path.exists(QuickReport.REPORTS_ROOT):
            os.mkdir(QuickReport.REPORTS_ROOT)

        report_name = str(date.today()) + '_' + '_'.join(self.title.split()) + '.tex'
        report_path = self._get_report_path()
        full_report_path = os.path.join(report_path, report_name)
        texfile = open(full_report_path, 'w')
        texfile.write(QuickReport.TEX_FILE_HEADER)
        texfile.write(r'\title{' + self.title + '}\n')
        texfile.write(r'''\begin{document}
\maketitle
''')
        for block in self.blocks:
            texfile.write(block)

        texfile.write(QuickReport.TEX_FILE_FOOTER)
        texfile.close()

        wd = os.getcwd()
        os.chdir(os.path.join(os.path.abspath(sys.path[0]), report_path))
        subprocess.call(['pdflatex', report_name])
        subprocess.call(['pdflatex', report_name]) # call the second time
        os.chdir(wd)

    def _get_report_path(self):
        report_name = str(date.today()) + '_' + '_'.join(self.title.split())
        return os.path.join(QuickReport.REPORTS_ROOT, report_name)

    def _get_images_path(self):
        return os.path.join(self._get_report_path(), QuickReport.IMAGES_DIR)


class QuickPresentation:
    """
    Class for making quick presentation
    """

    TEX_FILE_HEADER = r'''\documentclass{beamer}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{section in toc}[square]
\usetheme{Montpellier}
\beamersetuncovermixins{\opaqueness<1>{25}}{\opaqueness<2->{15}}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{tikz}
\usetikzlibrary{intersections}
\usepackage{epstopdf}
\usepackage[export]{adjustbox}
\usepackage[font={footnotesize}, labelfont={footnotesize}]{caption}
\usepackage[font={footnotesize}, labelfont={footnotesize}]{subcaption}
\usepackage[absolute,overlay]{textpos}
  \setlength{\TPHorizModule}{1mm}
  \setlength{\TPVertModule}{1mm}

\graphicspath{ {images/} }
\def\figurename{}

\defbeamertemplate*{title page}{customized}[1][]
{
  \begin{center}
  {\usebeamerfont{title}\usebeamercolor[fg]{title}\inserttitle}
  \\
  \vspace{10pt}
  \usebeamerfont{date}\insertdate
  \end{center}
  \bigskip
  \tableofcontents
}
\setbeamertemplate{footline}[frame number]
'''
    TEX_FILE_FOOTER = r'\end{document}'
    IMAGES_DIR = 'images'

    def __init__(self, presentation_root, title, date=None):
        self.presentation_root = presentation_root
        self.title = title
        self.date = date
        self.blocks = []
        if not os.path.exists(self.presentation_root):
            os.mkdir(self.presentation_root)
        report_path = self._get_report_path()
#        if os.path.exists(report_path):
#            shutil.rmtree(report_path)
        os.mkdir(report_path)
        os.mkdir(os.path.join(report_path, QuickPresentation.IMAGES_DIR))

    @classmethod
    def from_config(cls, title, date=None):
        with open(os.path.expanduser('~/.comsdk/config_research.json', 'r')) as f:
            conf = json.load(f)
        return QuickPresentation(conf['MEETINGS_PATH'], title, date)

    def print_out(self):
        if self.date is not None:
            tex_date = '{} {}, {}'.format(calendar.month_name[self.date[1]], self.date[2], self.date[0])
        else:
            tex_date = r'\today'

        report_name = self._get_path_date() + '_' + '_'.join(self.title.split()) + '.tex'
        report_path = self._get_report_path()
        full_report_path = os.path.join(report_path, report_name)
        texfile = open(full_report_path, 'w')
        texfile.write(QuickPresentation.TEX_FILE_HEADER)
        texfile.write(r'''\begin{document}
\setbeamertemplate{caption}{\raggedright\insertcaption\par}
\setbeamerfont{title}{size=\LARGE}
\setbeamerfont{date}{size=\footnotesize}
\title{''' + self.title + r'''} 
\date{''' + tex_date +  r'''}

\begin{frame}
\titlepage
\end{frame}
''')
        for block in self.blocks:
            texfile.write(block)

        texfile.write(QuickPresentation.TEX_FILE_FOOTER)
        texfile.close()

        wd = os.getcwd()
        os.chdir(os.path.join(os.path.abspath(sys.path[0]), report_path))
        subprocess.call(['pdflatex', report_name])
        subprocess.call(['pdflatex', report_name]) # call the second time
        os.chdir(wd)

    def _get_report_path(self):
        report_name = self._get_path_date() + '_' + '_'.join(self.title.split())
        return os.path.join(self.presentation_root, report_name)

    def _get_images_path(self):
        return os.path.join(self._get_report_path(), QuickPresentation.IMAGES_DIR)

    def _get_path_date(self):
        add_zero_if_needed = lambda num: str(num) if num >= 10 else '0{}'.format(num) 
        if self.date is not None:
            month = add_zero_if_needed(self.date[1])
            day = add_zero_if_needed(self.date[2])
            return '{}-{}-{}'.format(self.date[0], month, day)
        else:
            return str(date.today())
