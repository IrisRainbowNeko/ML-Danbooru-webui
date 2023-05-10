const MLD_PROGRESS_LABEL = 'mld_progress';

function mld_tagging() {
    var id = randomId();
    requestProgress(id,
        gradioApp().getElementById(MLD_PROGRESS_LABEL),
        null,
        function () {
        },
        function (progress) {
            gradioApp().getElementById(MLD_PROGRESS_LABEL).innerHTML = progress.textinfo
        })

    const argsToArray = args_to_array(arguments);
    argsToArray[0] = id;
    return argsToArray
}