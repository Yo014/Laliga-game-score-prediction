// La Liga Predictor Pro Client Application Orchestrator

document.addEventListener('DOMContentLoaded', () => {
    // App State
    const state = {
        teams: [],
        referees: [],
        activeTab: 'predictor-tab',
        activeSquadTeam: null,
        activeSquadPlayers: [],
        runningScripts: {},
        logPollingIntervals: {}
    };

    // DOM Elements
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    
    const homeTeamSelect = document.getElementById('home_team');
    const awayTeamSelect = document.getElementById('away_team');
    const refereeSelect = document.getElementById('referee');
    const homeRestDays = document.getElementById('home_rest_days');
    const awayRestDays = document.getElementById('away_rest_days');
    const homeRestVal = document.getElementById('home_rest_val');
    const awayRestVal = document.getElementById('away_rest_val');
    const predictionForm = document.getElementById('prediction-form');
    
    const predictionResultContainer = document.getElementById('prediction-result-container');
    const modelAccuracyBadge = document.getElementById('model-accuracy-badge');
    
    // Tab switching logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    function switchTab(tabId) {
        state.activeTab = tabId;
        
        // Update nav active states
        navItems.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update tab visibility
        tabContents.forEach(content => {
            if (content.id === tabId) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        // Update headers dynamically
        if (tabId === 'predictor-tab') {
            pageTitle.innerText = "Match Predictor";
            pageSubtitle.innerText = "Formulate matchups and simulate outcomes using local historical features and real-time squad health metrics.";
        } else if (tabId === 'squad-tab') {
            pageTitle.innerText = "Squad Health & Injury Manager";
            pageSubtitle.innerText = "Monitor player availability, manage active injuries, and see direct impacts on team competitiveness.";
            loadTeamsData(); // Refresh teams list
        } else if (tabId === 'pipeline-tab') {
            pageTitle.innerText = "Data & Machine Learning Pipeline";
            pageSubtitle.innerText = "Seed datasets, run advanced feature extraction, and retrain the prediction neural core.";
        }
    }

    // Sync Slider values in GUI
    homeRestDays.addEventListener('input', (e) => {
        homeRestVal.innerText = e.target.value;
    });
    awayRestDays.addEventListener('input', (e) => {
        awayRestVal.innerText = e.target.value;
    });

    // Initialize App Data
    async function initializeApp() {
        try {
            showNotification("System Boot", "Initializing ML models and loading historical records...", "info");
            
            // 1. Fetch Teams
            await loadTeamsData();
            
            // 2. Fetch Referees
            await loadRefereesData();
            
            // 3. Fetch Model Accuracy Status
            await loadModelStatus();
            
        } catch (err) {
            console.error("Initialization failed:", err);
            showNotification("Initialization Error", "Could not query data from backend server.", "danger");
        }
    }

    async function loadTeamsData() {
        try {
            const res = await fetch('/api/teams');
            const data = await res.json();
            
            if (data.success) {
                state.teams = data.teams;
                
                // Clear and rebuild selects
                const homeVal = homeTeamSelect.value;
                const awayVal = awayTeamSelect.value;
                
                homeTeamSelect.innerHTML = '<option value="" disabled selected>Select Home Team</option>';
                awayTeamSelect.innerHTML = '<option value="" disabled selected>Select Away Team</option>';
                
                // Sort teams alphabetically
                const sortedTeams = [...state.teams].sort((a, b) => a.Team.localeCompare(b.Team));
                
                sortedTeams.forEach(t => {
                    const optH = document.createElement('option');
                    optH.value = t.Team;
                    optH.innerText = t.Team;
                    homeTeamSelect.appendChild(optH);

                    const optA = document.createElement('option');
                    optA.value = t.Team;
                    optA.innerText = t.Team;
                    awayTeamSelect.appendChild(optA);
                });
                
                // Restore values if still available
                if (homeVal) homeTeamSelect.value = homeVal;
                if (awayVal) awayTeamSelect.value = awayVal;

                // Populate Squad Health list overview
                populateTeamHealthTable(sortedTeams);
            } else {
                showNotification("CSV Load Error", data.error, "danger");
            }
        } catch (err) {
            console.error(err);
        }
    }

    async function loadRefereesData() {
        try {
            const res = await fetch('/api/referees');
            const data = await res.json();
            
            if (data.success) {
                state.referees = data.referees;
                refereeSelect.innerHTML = '<option value="Unknown" selected>Select Referee (Optional)</option>';
                
                state.referees.forEach(ref => {
                    if (ref !== 'Unknown') {
                        const opt = document.createElement('option');
                        opt.value = ref;
                        opt.innerText = ref;
                        refereeSelect.appendChild(opt);
                    }
                });
            }
        } catch (err) {
            console.error(err);
        }
    }

    async function loadModelStatus() {
        try {
            const res = await fetch('/api/model-status');
            const data = await res.json();
            
            if (data.success) {
                const statValDiv = modelAccuracyBadge.closest('.quick-stat');
                if (data.trained && data.accuracy > 0) {
                    const accVal = (data.accuracy * 100).toFixed(1);
                    modelAccuracyBadge.innerText = `${accVal}%`;
                    modelAccuracyBadge.style.color = "var(--color-gold)";
                    if (statValDiv) {
                        statValDiv.setAttribute('title', `Prediction model trained with evaluated accuracy of ${accVal}%.`);
                    }
                } else {
                    modelAccuracyBadge.innerText = "0.0%";
                    modelAccuracyBadge.style.color = "var(--accent-danger)";
                    if (statValDiv) {
                        statValDiv.setAttribute('title', "Model accuracy not calculated yet. Run the Machine Learning Pipeline (Step 5) to retrain the core and compute model accuracy!");
                    }
                    
                    // Alert user with a descriptive toast
                    setTimeout(() => {
                        showNotification(
                            "Model Untrained", 
                            "The predictive model accuracy hasn't been calculated yet. Run the ML Pipeline (Step 5: Train Prediction Model) to train the core and compute accuracy!", 
                            "info"
                        );
                    }, 1500);
                }
            }
        } catch (err) {
            console.error("Could not fetch model status:", err);
            modelAccuracyBadge.innerText = "Error";
        }
    }

    // Populate the list of teams on the Roster tab
    function populateTeamHealthTable(teams) {
        const tbody = document.getElementById('team-health-list');
        tbody.innerHTML = '';

        teams.forEach(t => {
            const tr = document.createElement('tr');
            tr.dataset.team = t.Team;
            
            // Classify health status for color badge
            let healthColor = 'var(--color-home-win)'; // healthy green
            if (t.Missing_Impact_Pct > 15 || t.Missing_Key_Players >= 3) {
                healthColor = 'var(--accent-danger)'; // high impact red
            } else if (t.Missing_Impact_Pct > 5 || t.Missing_Key_Players >= 1) {
                healthColor = 'var(--color-draw)'; // moderate yellow
            }

            if (state.activeSquadTeam === t.Team) {
                tr.classList.add('active');
            }

            tr.innerHTML = `
                <td>
                    <div class="team-health-cell">
                        <span class="health-status-indicator" style="background-color: ${healthColor}; box-shadow: 0 0 6px ${healthColor};"></span>
                        <span>${t.Team}</span>
                    </div>
                </td>
                <td style="font-family: var(--font-mono); text-align:center;">${t.Total_Injured}</td>
                <td style="font-family: var(--font-mono); text-align:center;">${t.Missing_Key_Players}</td>
                <td style="font-family: var(--font-mono); font-weight: 600;">${t.Missing_Impact_Pct}%</td>
            `;

            tr.addEventListener('click', () => {
                // Remove active class from previous
                document.querySelectorAll('#team-health-list tr').forEach(row => row.classList.remove('active'));
                tr.classList.add('active');
                
                loadTeamRoster(t.Team);
            });

            tbody.appendChild(tr);
        });
    }

    // Load detailed squad roster for editing
    async function loadTeamRoster(teamName) {
        state.activeSquadTeam = teamName;
        
        // Hide placeholder, show loader/content
        document.getElementById('squad-editor-placeholder').classList.add('hidden');
        const contentDiv = document.getElementById('squad-editor-content');
        contentDiv.classList.remove('hidden');
        
        document.getElementById('active-squad-name').innerHTML = `Managing <strong>${teamName}</strong> player profiles`;
        
        const listBody = document.getElementById('squad-players-list');
        listBody.innerHTML = '<tr><td colspan="7" class="text-muted" style="text-align: center; padding: 40px;"><i class="fa-solid fa-spinner fa-spin"></i> Retrieving player details...</td></tr>';
        
        try {
            const res = await fetch(`/api/squad/${teamName}`);
            const data = await res.json();
            
            if (data.success) {
                state.activeSquadPlayers = data.players;
                document.getElementById('player-count').innerText = `${data.players.length} Players Listed`;
                
                listBody.innerHTML = '';
                
                data.players.forEach(p => {
                    const tr = document.createElement('tr');
                    
                    const isInjured = parseInt(p.Injuries) > 0;
                    
                    tr.innerHTML = `
                        <td class="player-name-cell">${p.Player}</td>
                        <td style="font-family: var(--font-mono); text-align:center;">${p.Appearances}</td>
                        <td style="font-family: var(--font-mono); text-align:center;">${p['Goals Scored'] || p.Gls || 0}</td>
                        <td>
                            <label class="status-toggle">
                                <input type="checkbox" class="injury-toggle" data-player="${p.Player}" ${isInjured ? 'checked' : ''}>
                                <span class="toggle-slider"></span>
                            </label>
                        </td>
                        <td>
                            <input type="text" class="table-input player-day-injured" data-player="${p.Player}" value="${p['Day Injured'] || ''}" placeholder="DD/MM/YYYY" ${!isInjured ? 'disabled' : ''}>
                        </td>
                        <td>
                            <input type="number" class="table-input player-games-out" data-player="${p.Player}" min="0" value="${p['Missed Games'] || 0}" ${!isInjured ? 'disabled' : ''}>
                        </td>
                        <td>
                            <input type="text" class="table-input player-expected-return" data-player="${p.Player}" value="${p['Expected Return'] || ''}" placeholder="DD/MM/YYYY or Unknown" ${!isInjured ? 'disabled' : ''}>
                        </td>
                    `;
                    
                    // Add listener to toggles
                    const toggleInput = tr.querySelector('.injury-toggle');
                    const dayInjInput = tr.querySelector('.player-day-injured');
                    const gamesOutInput = tr.querySelector('.player-games-out');
                    const expRetInput = tr.querySelector('.player-expected-return');
                    
                    toggleInput.addEventListener('change', (e) => {
                        const checked = e.target.checked;
                        dayInjInput.disabled = !checked;
                        gamesOutInput.disabled = !checked;
                        expRetInput.disabled = !checked;
                        
                        if (checked) {
                            // Populate logical defaults
                            const todayStr = getTodayFormattedString();
                            dayInjInput.value = todayStr;
                            gamesOutInput.value = 1;
                            expRetInput.value = "Unknown";
                        } else {
                            dayInjInput.value = '';
                            gamesOutInput.value = 0;
                            expRetInput.value = '';
                        }
                    });
                    
                    listBody.appendChild(tr);
                });
            } else {
                listBody.innerHTML = `<tr><td colspan="7" class="text-danger" style="text-align: center; padding: 40px;"><i class="fa-solid fa-circle-exclamation"></i> Error: ${data.error}</td></tr>`;
            }
        } catch (err) {
            console.error(err);
            listBody.innerHTML = `<tr><td colspan="7" class="text-danger" style="text-align: center; padding: 40px;"><i class="fa-solid fa-circle-exclamation"></i> API request failed. Check server connection.</td></tr>`;
        }
    }

    // Save squad edits
    document.getElementById('save-squad-btn').addEventListener('click', async () => {
        if (!state.activeSquadTeam) return;
        
        const saveBtn = document.getElementById('save-squad-btn');
        const origHtml = saveBtn.innerHTML;
        saveBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving changes...';
        saveBtn.disabled = true;
        
        const updatedPlayers = [];
        
        // Read input rows
        const toggles = document.querySelectorAll('.injury-toggle');
        toggles.forEach(tog => {
            const playerName = tog.dataset.player;
            const parentRow = tog.closest('tr');
            
            const isInjured = tog.checked ? 1 : 0;
            const dayInjured = parentRow.querySelector('.player-day-injured').value.trim();
            const gamesOut = parseInt(parentRow.querySelector('.player-games-out').value) || 0;
            const expectedReturn = parentRow.querySelector('.player-expected-return').value.trim();
            
            updatedPlayers.push({
                Player: playerName,
                Injuries: isInjured,
                "Day Injured": dayInjured,
                "Missed Games": gamesOut,
                "Expected Return": expectedReturn
            });
        });
        
        try {
            const res = await fetch(`/api/squad/${state.activeSquadTeam}/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ players: updatedPlayers })
            });
            
            const data = await res.json();
            
            if (data.success) {
                showNotification("Success", `Roster changes saved for ${state.activeSquadTeam}. Squad health recalculated.`, "success");
                // Reload data
                await loadTeamsData();
                await loadTeamRoster(state.activeSquadTeam);
            } else {
                showNotification("Update Failed", data.error, "danger");
            }
        } catch (err) {
            console.error(err);
            showNotification("Connection Failure", "Could not send edits to server.", "danger");
        } finally {
            saveBtn.innerHTML = origHtml;
            saveBtn.disabled = false;
        }
    });

    // Helper: returns DD/MM/YYYY string for today
    function getTodayFormattedString() {
        const today = new Date();
        const yyyy = today.getFullYear();
        let mm = today.getMonth() + 1; // Months start at 0!
        let dd = today.getDate();

        if (dd < 10) dd = '0' + dd;
        if (mm < 10) mm = '0' + mm;

        return dd + '/' + mm + '/' + yyyy;
    }

    // Prediction Form Submission Handler
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const predictBtn = document.getElementById('predict-btn');
        const origHtml = predictBtn.innerHTML;
        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Matchup...';
        predictBtn.disabled = true;

        const payload = {
            home_team: homeTeamSelect.value,
            away_team: awayTeamSelect.value,
            home_rest_days: parseInt(homeRestDays.value),
            away_rest_days: parseInt(awayRestDays.value),
            b365h: parseFloat(document.getElementById('b365h').value),
            b365d: parseFloat(document.getElementById('b365d').value),
            b365a: parseFloat(document.getElementById('b365a').value),
            referee: refereeSelect.value
        };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await res.json();
            
            if (data.success) {
                renderPredictionResults(data.data);
            } else {
                showNotification("Prediction Error", data.error, "danger");
            }
        } catch (err) {
            console.error(err);
            showNotification("Inference Failed", "Model calculation timed out or crashed.", "danger");
        } finally {
            predictBtn.innerHTML = origHtml;
            predictBtn.disabled = false;
        }
    });

    // Display prediction outcome values on screen
    function renderPredictionResults(res) {
        // Hide placeholder, show actual results panel
        predictionResultContainer.querySelector('.result-placeholder').classList.add('hidden');
        const actualContainer = predictionResultContainer.querySelector('.actual-results');
        actualContainer.classList.remove('hidden');

        // Set outcome title
        const outcomeBadge = document.getElementById('res-prediction-outcome');
        outcomeBadge.innerText = res.prediction.toUpperCase();
        
        // Color code outcome badge background
        if (res.prediction === "Home Win") {
            outcomeBadge.style.color = "var(--color-home-win)";
        } else if (res.prediction === "Away Win") {
            outcomeBadge.style.color = "var(--color-away-win)";
        } else {
            outcomeBadge.style.color = "var(--color-draw)";
        }

        // Set probability bar widths and texts
        const homePct = (res.probabilities.home * 100).toFixed(1);
        const drawPct = (res.probabilities.draw * 100).toFixed(1);
        const awayPct = (res.probabilities.away * 100).toFixed(1);

        const homeBar = document.getElementById('prob-home-bar');
        const drawBar = document.getElementById('prob-draw-bar');
        const awayBar = document.getElementById('prob-away-bar');

        // Apply width and text content
        homeBar.style.width = `${homePct}%`;
        document.getElementById('prob-home-val').innerText = `${homePct}%`;
        
        drawBar.style.width = `${drawPct}%`;
        document.getElementById('prob-draw-val').innerText = `${drawPct}%`;
        
        awayBar.style.width = `${awayPct}%`;
        document.getElementById('prob-away-val').innerText = `${awayPct}%`;

        // Set Metric values
        document.getElementById('metric-home-value').innerText = `€${(res.home_squad_value / 1e6).toFixed(1)}M`;
        document.getElementById('metric-away-value').innerText = `€${(res.away_squad_value / 1e6).toFixed(1)}M`;

        document.getElementById('metric-home-offense').innerText = res.home_expected_offense.toFixed(2);
        document.getElementById('metric-away-offense').innerText = res.away_expected_offense.toFixed(2);

        // Sidelined Player health summary details
        document.getElementById('home-team-name-badge').innerText = res.home_team;
        document.getElementById('home-health-info').innerHTML = `
            Missing <strong class="text-gold">${res.home_missing_key}</strong> key players 
            (<span class="text-danger">${res.home_missing_impact.toFixed(1)}%</span> playing impact, 
            <span class="text-danger">${res.home_missing_goals.toFixed(1)}%</span> goals)
        `;

        document.getElementById('away-team-name-badge').innerText = res.away_team;
        document.getElementById('away-health-info').innerHTML = `
            Missing <strong class="text-gold">${res.away_missing_key}</strong> key players 
            (<span class="text-danger">${res.away_missing_impact.toFixed(1)}%</span> playing impact, 
            <span class="text-danger">${res.away_missing_goals.toFixed(1)}%</span> goals)
        `;

        showNotification("Match Analyzed", `Prediction compiled successfully for ${res.home_team} vs ${res.away_team}.`, "success");
    }

    // RUN SCRIPTS ORCHESTRATION PIPELINE
    const runScriptBtns = document.querySelectorAll('.run-script-btn');
    const terminalStdout = document.getElementById('terminal-stdout');
    const consoleTargetName = document.getElementById('console-target-name');

    runScriptBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            const scriptName = btn.getAttribute('data-script');
            executePipelineScript(scriptName);
        });
    });

    async function executePipelineScript(scriptName) {
        // Toggle UI button
        const pill = document.getElementById(`status-${scriptName}`);
        pill.innerText = "Running";
        pill.className = "status-pill running";
        
        terminalStdout.innerText = `[Engine] Launching subprocess: ${scriptName}...\n`;
        consoleTargetName.innerText = scriptName;

        try {
            const res = await fetch('/api/run-script', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script: scriptName })
            });

            const data = await res.json();
            
            if (data.success) {
                showNotification("Process Launched", `${scriptName} is executing in the background.`, "info");
                // Start polling logs
                startLogPolling(scriptName);
            } else {
                pill.innerText = "Failed";
                pill.className = "status-pill failed";
                terminalStdout.innerText += `[Engine Error] Failed to launch: ${data.error}\n`;
                showNotification("Launch Failed", data.error, "danger");
            }
        } catch (err) {
            console.error(err);
            pill.innerText = "Failed";
            pill.className = "status-pill failed";
            terminalStdout.innerText += `[Engine Error] Connection crashed while executing script.\n`;
        }
    }

    function startLogPolling(scriptName) {
        // Clear old interval if exists
        if (state.logPollingIntervals[scriptName]) {
            clearInterval(state.logPollingIntervals[scriptName]);
        }

        const pollLogs = async () => {
            try {
                const res = await fetch(`/api/script-logs/${scriptName}`);
                const data = await res.json();
                
                if (data.success) {
                    // Update Terminal output
                    if (consoleTargetName.innerText === scriptName) {
                        terminalStdout.innerText = data.logs || "Awaiting logs...";
                        // Auto scroll console
                        const consoleWrapper = terminalStdout.closest('.console-body-wrapper');
                        consoleWrapper.scrollTop = consoleWrapper.scrollHeight;
                    }

                    // Check if complete
                    const pill = document.getElementById(`status-${scriptName}`);
                    if (data.status === 'completed') {
                        pill.innerText = "Completed";
                        pill.className = "status-pill completed";
                        clearInterval(state.logPollingIntervals[scriptName]);
                        showNotification("Process Finished", `${scriptName} has completed successfully.`, "success");
                        
                        // If it is train_model.py, grab model accuracy
                        if (scriptName === 'train_model.py') {
                            extractNewAccuracy(data.logs);
                        }
                    } else if (data.status === 'failed') {
                        pill.innerText = "Failed";
                        pill.className = "status-pill failed";
                        clearInterval(state.logPollingIntervals[scriptName]);
                        showNotification("Process Failed", `${scriptName} exited with errors.`, "danger");
                    }
                }
            } catch (err) {
                console.error("Log polling error:", err);
            }
        };

        // Poll every 1 second
        pollLogs(); // Run immediately
        state.logPollingIntervals[scriptName] = setInterval(pollLogs, 1000);
    }

    function extractNewAccuracy(logs) {
        // Search logs for "Overall Accuracy: XX.XX%"
        const match = logs.match(/Overall Accuracy:\s*([0-9\.]+)%/i);
        if (match && match[1]) {
            const accVal = parseFloat(match[1]).toFixed(1);
            modelAccuracyBadge.innerText = `${accVal}%`;
            modelAccuracyBadge.style.color = "var(--color-gold)";
            const statValDiv = modelAccuracyBadge.closest('.quick-stat');
            if (statValDiv) {
                statValDiv.setAttribute('title', `Prediction model trained with evaluated accuracy of ${accVal}%.`);
            }
            showNotification("Accuracy Restructured", `Prediction core retrained. Accuracy recorded: ${accVal}%`, "success");
        }
    }

    // Clear console terminal
    document.getElementById('clear-console-btn').addEventListener('click', () => {
        terminalStdout.innerText = "Console buffer flushed. Ready for process signals.";
    });

    // TOAST NOTIFICATIONS
    function showNotification(title, message, type = 'success') {
        const toast = document.getElementById('notification-toast');
        const icon = toast.querySelector('.info-icon');
        const titleEl = document.getElementById('notif-title');
        const msgEl = document.getElementById('notif-message');

        titleEl.innerText = title;
        msgEl.innerText = message;

        // Reset classes
        toast.className = "notification";
        icon.className = "fa-solid info-icon";

        if (type === 'success') {
            toast.style.borderColor = "rgba(16, 185, 129, 0.35)";
            icon.classList.add('fa-circle-check', 'text-success');
        } else if (type === 'danger') {
            toast.style.borderColor = "rgba(239, 68, 68, 0.35)";
            icon.classList.add('fa-circle-exclamation', 'text-danger');
        } else {
            // Info
            toast.style.borderColor = "rgba(226, 184, 66, 0.35)";
            icon.classList.add('fa-circle-info', 'text-gold');
        }

        // Remove hidden
        toast.classList.remove('hidden');

        // Autohide after 5 seconds
        if (state.toastTimeout) clearTimeout(state.toastTimeout);
        state.toastTimeout = setTimeout(() => {
            toast.classList.add('hidden');
        }, 5000);
    }

    // Close notification btn
    document.getElementById('notif-close-btn').addEventListener('click', () => {
        document.getElementById('notification-toast').classList.add('hidden');
    });

    // Initialize application
    initializeApp();
});
