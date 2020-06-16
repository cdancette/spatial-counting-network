import urwid
import os
import subprocess

# choices = "Chapman Cleese Gilliam Idle Jones Palin".split()

def get_job_info(job_id):
    res = subprocess.run(
        f"scontrol show job {job_id}",
        shell=True,
        stdout=subprocess.PIPE,
    )
    return res.stdout.decode()

def get_runninng_jobs():
    res = subprocess.run(
        "squeue -u `whoami` -o '%i|%P|%t|%M|%D|%R|%j' --sort=i",
        shell=True,
        stdout=subprocess.PIPE,
    )
    columns = ["jobid", "partition", "st", "time", "nodes", "nodelist", "name"]
    results = []
    for line in res.stdout.decode().split():
        line = line.split("|")
        results.append(line)
    return results


def get_list_body():
    results = get_runninng_jobs()
    choices = [f"{r[0]:8} {r[2]} {r[3]:10} {r[5]:8} {r[6]}" for r in results]
    header = choices[0]
    choices = choices[1:]  # names
    results = results[1:]
    body = [urwid.Text("Runnning jobs"), urwid.Divider()]
    body.append(urwid.Text(header))
    for i, c in enumerate(choices):
        button = urwid.Button(c)
        urwid.connect_signal(button, "click", item_chosen, results[i][0])
        body.append(urwid.AttrMap(button, None, focus_map="reversed"))
    return body


def menu():
    body = get_list_body()
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))


def close_submenu(button):
    main.original_widget = main.original_widget[0]


def cancel_job(button, job_id):
    res = subprocess.run(f"scancel  {job_id}", shell=True, stdout=subprocess.PIPE,)
    main.original_widget = main.original_widget[0]


def submenu(job_id):
    # return urwid.Text("Overlay !!! \n")))

    back = urwid.Button("Back")
    cancel = urwid.Button(f"/!\\ Cancel job {job_id}")
    body = [
        back,
        cancel
    ]

    urwid.connect_signal(back, "click", close_submenu)
    urwid.connect_signal(cancel, "click", cancel_job, job_id)

    return urwid.LineBox(
        urwid.Filler(
            urwid.Pile(
                [
                    urwid.BoxAdapter(
                        urwid.ListBox(urwid.SimpleFocusListWalker(body)), height=2,
                    ),
                    # urwid.Text(get_job_info(job_id)),
                ]
            )
        )
    )

    return urwid.LineBox(
        urwid.Filler(
            urwid.BoxAdapter(
                urwid.ListBox(urwid.SimpleFocusListWalker(body)), height=10,
            )
        )
    )


def item_chosen(button, choice):
    done = urwid.Button("Ok")
    urwid.connect_signal(done, "click", exit_program)
    main.original_widget = urwid.Overlay(
        submenu(choice),
        main.original_widget,
        align="center",
        width=("relative", 30),
        valign="middle",
        height=("relative", 30),
        min_width=20,
        min_height=9,
    )


def refresh_main_list(mainlist):
    mainlist.body[:] = get_list_body()


def exit_program(button):
    raise urwid.ExitMainLoop()


main_list = menu()
main = urwid.Padding(main_list, left=2, right=2)

# top = urwid.Overlay(
#     main,
#     urwid.SolidFill(),
#     align="center",
#     width=("relative", 60),
#     valign="middle",
#     height=("relative", 60),
#     min_width=20,
#     min_height=9,
# )


def update(loop, user_data):
    refresh_main_list(main_list)
    # loop.draw_screen()
    loop.set_alarm_in(20, update)


loop = urwid.MainLoop(main, palette=[("reversed", "standout", "")])
loop.set_alarm_in(4, update)
loop.run()

"""

def menu_button(caption, callback):
    button = urwid.Button(caption)
    urwid.connect_signal(button, 'click', callback)
    return urwid.AttrMap(button, None, focus_map='reversed')

def sub_menu(caption, choices):
    contents = menu(caption, choices)
    def open_menu(button):
        return top.open_box(contents)
    return menu_button([caption, u'...'], open_menu)

def menu(title, choices):
    body = [urwid.Text(title), urwid.Divider()]
    body.extend(choices)
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))

def item_chosen(button):
    response = urwid.Text([u'You chose ', button.label, u'\n'])
    done = menu_button(u'Ok', exit_program)
    top.open_box(urwid.Filler(urwid.Pile([response, done])))

def exit_program(button):
    raise urwid.ExitMainLoop()

menu_top = menu(u'Main Menu', [
    sub_menu(u'Applications', [
        sub_menu(u'Accessories', [
            menu_button(u'Text Editor', item_chosen),
            menu_button(u'Terminal', item_chosen),
        ]),
    ]),
    sub_menu(u'System', [
        sub_menu(u'Preferences', [
            menu_button(u'Appearance', item_chosen),
        ]),
        menu_button(u'Lock Screen', item_chosen),
    ]),
])


# class CascadingBoxes(urwid.WidgetPlaceholder):
#     max_box_levels = 4

#     def __init__(self, box):
#         super(CascadingBoxes, self).__init__(urwid.SolidFill(u'/'))
#         self.box_level = 0
#         self.open_box(box)

#     def open_box(self, box):
#         self.original_widget = urwid.Overlay(urwid.LineBox(box),
#             self.original_widget,
#             align='center', width=('relative', 80),
#             valign='middle', height=('relative', 80),
#             min_width=24, min_height=8,
#             left=self.box_level * 3,
#             right=(self.max_box_levels - self.box_level - 1) * 3,
#             top=self.box_level * 2,
#             bottom=(self.max_box_levels - self.box_level - 1) * 2)
#         self.box_level += 1

#     def keypress(self, size, key):
#         if key == 'esc' and self.box_level > 1:
#             self.original_widget = self.original_widget[0]
#             self.box_level -= 1
#         else:
#             return super(CascadingBoxes, self).keypress(size, key)

top = CascadingBoxes(menu_top)
urwid.MainLoop(top, palette=[('reversed', 'standout', '')]).run()
"""
