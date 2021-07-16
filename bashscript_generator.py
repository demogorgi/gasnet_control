import os
import sys

template_path = '/home/adi/Uni/SoSe21/Masterarbeit/cluster/' \
                'cdqn_template.sh'
destination_path = '/home/adi/Uni/SoSe21/Masterarbeit/cluster/'
update_steps = [1, 5, 20, 100, 500] #, 50, 100, 200, 500, 2000] #[200, 500, 2000, 5000]
epsilons = [1, 0.5, 0.25, 0.1] #[0.5, 0.25, 0.1, 0.05]
gradient_clippings = ['None', 1] #, 1.0] #, 1.0] #['None', 1.0, 10.0]
learning_rates = [1e-2] #[1e-1, 1e-2, 1e-3] #[1e-3, 1e-4, 1e-5, 1e-6]
layers = [(20,), (50,), (100,), (250,)]
epsilon_decay = True
end_epsilon = 0.001
learning_rate_decay = True
end_learning_rate = 1e-5
if len(sys.argv) > 1:
    run = int(sys.argv[1])
else:
    run = -1

if run >= 0:
    template_path = template_path.replace(".sh", "_valrun.sh")

if __name__ == '__main__':
    if 'cdqn' in template_path:
        use_cdqn = True
    else:
        use_cdqn = False
    drop_path = destination_path + "cdqn_scripts_reducedinput/"
    try:
        os.makedirs(drop_path)
    except FileExistsError:
        pass

    with open(template_path, 'r') as template_file:
        template_data = template_file.read()

    for rate in learning_rates:
        rate_string = "rate"
        rate_string += '{:.0e}'.format(rate).replace('0', '')
        if learning_rate_decay:
            rate_string += "to"
            rate_string += '{:.0e}'.format(end_learning_rate).replace('0', '')
        rate_path = drop_path + rate_string + "/"
        try:
            os.makedirs(rate_path)
        except FileExistsError:
            pass

        for clip in gradient_clippings:
            if type(clip) == float:
                clip_string = f"clip{int(clip)}"
            else:
                clip_string = f"clip{clip}"
            clip_path = rate_path + clip_string + "/"
            try:
                os.makedirs(clip_path)
            except FileExistsError:
                pass

            for net in layers:
                net_string = "net"
                for layer in net:
                    net_string += f"{layer}_"
                net_string = net_string[:-1]
                net_path = clip_path + net_string + "/"
                try:
                    os.makedirs(net_path)
                except FileExistsError:
                    pass

                for epsilon in epsilons:
                    epsilon_string = f"epsilon{str(epsilon).replace('.', '')}"
                    epsilon_decay_string = ""
                    if epsilon_decay:
                        epsilon_decay_string = "to"
                        epsilon_decay_string += str(end_epsilon).replace(
                            '.', '')
                        epsilon_string += epsilon_decay_string
                    epsilon_path = net_path + epsilon_string + "/"
                    try:
                        os.makedirs(epsilon_path)
                    except FileExistsError:
                        pass

                    for steps in update_steps:
                        bashfile_content = template_data
                        bashfile_content = bashfile_content.replace(
                            "[nupdate]",
                            str(steps))
                        bashfile_content = bashfile_content.replace(
                            "[start_eps]",
                            str(epsilon))
                        bashfile_content = bashfile_content.replace(
                            "[gradient]",
                            str(clip))
                        if learning_rate_decay:
                            bashfile_content = bashfile_content.replace(
                                "[lr]",
                                rate_string.replace('rate', ""))
                        else:
                            bashfile_content = bashfile_content.replace(
                                "[lr]",
                                str(rate))
                        layer_string = ""
                        for layer in net:
                            layer_string += str(layer) + "-"
                        layer_string = layer_string[:-1]
                        bashfile_content = bashfile_content.replace(
                            "[layers]",
                            layer_string)
                        bashfile_content = bashfile_content.replace(
                            "[eps_decay]",
                            "T" if epsilon_decay else "F")
                        bashfile_content = bashfile_content.replace(
                            "[end_eps]",
                            str(end_epsilon))
                        bashfile_content = bashfile_content.replace(
                            "[start_eps_desc]",
                            str(epsilon).replace(".", ""))
                        if epsilon_decay:
                            bashfile_content = bashfile_content.replace(
                                "[eps_decay_desc]",
                                epsilon_decay_string)
                        else:
                            bashfile_content = bashfile_content.replace(
                                "[eps_decay_desc]",
                                "")
                        if learning_rate_decay:
                            bashfile_content = bashfile_content.replace(
                                "[lr_desc]",
                                rate_string.replace('rate', ""))
                        else:
                            bashfile_content = bashfile_content.replace(
                                "[lr_desc]",
                                "{:.0e}".format(rate).replace('0', ''))
                        bashfile_content = bashfile_content.replace(
                            "[gradient_desc]",
                            'None' if clip == 'None' else str(int(clip)))
                        if run >= 0:
                            bashfile_content = bashfile_content.replace(
                                "[no. run]",
                                str(run))

                        bashfile_name = f"dqn_net{layer_string}_{steps}updates"
                        bashfile_name += f"_{str(epsilon).replace('.', '')}eps"
                        bashfile_name += f"{epsilon_decay_string}"

                        if learning_rate_decay:
                            bashfile_name += "_" + rate_string
                        else:
                            bashfile_name += f"_rate"
                            bashfile_name += '{:.0e}'.format(rate).replace(
                                '0', '')
                        if clip == 'None':
                            bashfile_name += f"clipNone"
                        else:
                            bashfile_name += f"clip{str(int(clip))}"
                        if '200k' in template_path:
                            bashfile_name += f"_200k"
                        if run >= 0:
                            bashfile_name += f"_run{run}"
                        bashfile_name += f".sh"

                        if use_cdqn:
                            bashfile_name = bashfile_name.replace(
                                'dqn', 'cdqn'
                            )
                        with open(epsilon_path + bashfile_name, 'w+') as bfile:
                            bfile.write(bashfile_content)
                        print(f"Written {epsilon_path + bashfile_name}")