import arguments
import cavia
import maml
import cva
import twoplayergame
import two_stage_cva


if __name__ == '__main__':

    args = arguments.parse_args()

    if args.maml:
        logger = maml.run(args, log_interval=1000, rerun=True)
    if args.vanilla:
        logger = two_stage_cva.run(args, log_interval=1000, rerun=True)
    elif args.cva:
        logger = cva.run(args, log_interval=2000, rerun=True)
    else:
        logger = cavia.run(args, log_interval=1000, rerun=True)
