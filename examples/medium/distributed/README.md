# Development flow

## Part one - local development
1. Come up with the complete flow, and a single script to run it
2. Make sure parameters can be configured on the command line, specifically:
    - Input / output / model paths
    - The number of processes / workers for data loaders
    - Distribution and number of used nodes (allow a single node as well)
    - Hyperparameters - batch size, learning rate, number of epochs etc
3. Run locally
    - With / without distribution (of size 1)
    - Check CPU, RAM, GPU and GPU RAM usage
        - Figure out a good balance between batch size and the learning rate until you reach a bottleneck
    - Save the model to the "state" directory every few cycles (e.g. every min or two, assuming saving is quick)
        - PyTorch: save it once on the beginning of the loop (to avoid many messages from the debugger `smdebug`)
4. Make sure running the entire flow again continue from where it stopped

## Part two - moving remotely
5. Update the code to support simple-sagemaker
    - The training script + running script
    - TBD: Tutorial / post on tihs
6. Test locally using "local mode" "--it local" to make sure everything works
    - Note: not everything is supported (TBD e.g.), but you may be able to find a few bugs quicker
7. Test remotely
    - Start with "--no_spot" to accelerate iterations until you're ready
    - Check CPU, RAM, GPU and GPU RAM usage
        - Figure out a good balance between batch size and the learning rate until you reach a bottleneck

## Part three - profit!
8. Run remotely
    - Make sure to remove "--no_spot"
9. Hyperparameters tuning
10. Debugging