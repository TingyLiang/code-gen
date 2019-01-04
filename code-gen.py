# coding=utf-8

from string import Template


class CodeGenerator:
    def __init__(self):
        return self

    def gen_code(self):
        filePath = '../template'
        class_file = open(filePath, 'w')

        mycode = []

        # 加载模板文件
        template_file = open('high-level.tmpl', 'r')
        tmpl = Template(template_file.read())
        # 模板替换
        mycode.append(tmpl.substitute(
            CLASSNAME='DEFAULT',
            Class_Name='Default',
            En_name='mystruct',
            Type='int',
            Name='value'))

        # 将代码写入文件
        class_file.writelines(mycode)
        class_file.close()

        print('ok')

    def parse_source_code(self):
        path = '../config/tf-cluster-keras.py'


if __name__ == '__main__':
    build = CodeGenerator()
    build.gen_code()
