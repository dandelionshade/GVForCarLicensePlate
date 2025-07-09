# coding=gbk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
root = ttk.Window()
style = ttk.Style()
theme_names = style.theme_names()#以列表的形式返回多个主题名
theme_selection = ttk.Frame(root, padding=(10, 10, 10, 0))
theme_selection.pack(fill=X, expand=YES)
lbl = ttk.Label(theme_selection, text="选择主题:")
theme_cbo = ttk.Combobox(
        master=theme_selection,
        text=style.theme.name,
        values=theme_names,
)
theme_cbo.pack(padx=10, side=RIGHT)
theme_cbo.current(theme_names.index(style.theme.name))
lbl.pack(side=RIGHT)
def change_theme(event):
    theme_cbo_value = theme_cbo.get()
    style.theme_use(theme_cbo_value)
    theme_selected.configure(text=theme_cbo_value)
    theme_cbo.selection_clear()
theme_cbo.bind('<<ComboboxSelected>>', change_theme)
theme_selected = ttk.Label(
        master=theme_selection,
        text="litera",
        font="-size 24 -weight bold"
)
theme_selected.pack(side=LEFT)
root.mainloop()
