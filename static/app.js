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
        logPollingIntervals: {},
        priceChartInstance: null
    };

    // DOM Elements
    const navItems = document.querySelectorAll('.nav-item, .top-nav-item, [data-tab]');
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

    // Polymarket Outcome Buttons
    const btnHomeOutcome = document.querySelector('.outcome-button.home-win-btn');
    const btnDrawOutcome = document.querySelector('.outcome-button.draw-btn');
    const btnAwayOutcome = document.querySelector('.outcome-button.away-win-btn');
    
    // Tab switching logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            if (targetTab) {
                switchTab(targetTab);
            }
        });
    });

    // Market Question Dynamic Update
    if (homeTeamSelect && awayTeamSelect) {
        homeTeamSelect.addEventListener('change', updateMarketQuestion);
        awayTeamSelect.addEventListener('change', updateMarketQuestion);
    }

    function updateMarketQuestion() {
        const homeTeam = (homeTeamSelect && homeTeamSelect.value) ? homeTeamSelect.options[homeTeamSelect.selectedIndex]?.text : "[Home Team]";
        const awayTeam = (awayTeamSelect && awayTeamSelect.value) ? awayTeamSelect.options[awayTeamSelect.selectedIndex]?.text : "[Away Team]";
        
        const questionHome = document.getElementById('question-home-team');
        const questionAway = document.getElementById('question-away-team');
        if (questionHome) questionHome.innerText = homeTeam;
        if (questionAway) questionAway.innerText = awayTeam;
    }

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

        // Update headers dynamically (safe check for missing headers in reskinned designs)
        if (pageTitle) {
            if (tabId === 'predictor-tab') pageTitle.innerText = "Match Predictor";
            else if (tabId === 'squad-tab') pageTitle.innerText = "Squad Health & Injury Manager";
            else if (tabId === 'pipeline-tab') pageTitle.innerText = "Data & Machine Learning Pipeline";
        }
        if (pageSubtitle) {
            if (tabId === 'predictor-tab') pageSubtitle.innerText = "Formulate matchups and simulate outcomes using local historical features and real-time squad health metrics.";
            else if (tabId === 'squad-tab') pageSubtitle.innerText = "Monitor player availability, manage active injuries, and see direct impacts on team competitiveness.";
            else if (tabId === 'pipeline-tab') pageSubtitle.innerText = "Seed datasets, run advanced feature extraction, and retrain the prediction neural core.";
        }
        
        if (tabId === 'squad-tab') {
            loadTeamsData(); // Refresh teams list
        }
    }

    // Sync Slider values in GUI
    if (homeRestDays && homeRestVal) {
        homeRestDays.addEventListener('input', (e) => {
            homeRestVal.innerText = e.target.value;
        });
    }
    if (awayRestDays && awayRestVal) {
        awayRestDays.addEventListener('input', (e) => {
            awayRestVal.innerText = e.target.value;
        });
    }

    // Sync Question and Outcome Buttons on Team Select change
    const questionHome = document.getElementById('question-home-team');
    const questionAway = document.getElementById('question-away-team');
    const outcomeHomeLabel = document.getElementById('outcome-home-label');
    const outcomeAwayLabel = document.getElementById('outcome-away-label');

    if (homeTeamSelect) {
        homeTeamSelect.addEventListener('change', () => {
            const name = homeTeamSelect.value || '[Home Team]';
            if (questionHome) questionHome.innerText = name;
            if (outcomeHomeLabel) outcomeHomeLabel.innerText = `${name} Win`;
        });
    }

    if (awayTeamSelect) {
        awayTeamSelect.addEventListener('change', () => {
            const name = awayTeamSelect.value || '[Away Team]';
            if (questionAway) questionAway.innerText = name;
            if (outcomeAwayLabel) outcomeAwayLabel.innerText = `${name} Win`;
        });
    }

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
            
            // 4. Render Initial Flat Price Chart
            renderInitialChart();
            
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
                const homeVal = homeTeamSelect ? homeTeamSelect.value : '';
                const awayVal = awayTeamSelect ? awayTeamSelect.value : '';
                
                if (homeTeamSelect) homeTeamSelect.innerHTML = '<option value="" disabled selected>Select Home Team</option>';
                if (awayTeamSelect) awayTeamSelect.innerHTML = '<option value="" disabled selected>Select Away Team</option>';
                
                // Sort teams alphabetically
                const sortedTeams = [...state.teams].sort((a, b) => a.Team.localeCompare(b.Team));
                
                sortedTeams.forEach(t => {
                    if (homeTeamSelect) {
                        const optH = document.createElement('option');
                        optH.value = t.Team;
                        optH.innerText = t.Team;
                        homeTeamSelect.appendChild(optH);
                    }

                    if (awayTeamSelect) {
                        const optA = document.createElement('option');
                        optA.value = t.Team;
                        optA.innerText = t.Team;
                        awayTeamSelect.appendChild(optA);
                    }
                });
                
                // Restore values if still available
                if (homeTeamSelect && homeVal) homeTeamSelect.value = homeVal;
                if (awayTeamSelect && awayVal) awayTeamSelect.value = awayVal;

                // Dynamically update the market question text
                updateMarketQuestion();

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
            
            if (data.success && refereeSelect) {
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
                const statValDiv = modelAccuracyBadge ? modelAccuracyBadge.closest('.quick-stat') : null;
                if (data.trained && data.accuracy > 0) {
                    const accVal = (data.accuracy * 100).toFixed(1);
                    if (modelAccuracyBadge) {
                        modelAccuracyBadge.innerText = `${accVal}%`;
                        modelAccuracyBadge.style.color = "var(--color-gold)";
                    }
                    if (statValDiv) {
                        statValDiv.setAttribute('title', `Prediction model trained with evaluated accuracy of ${accVal}%.`);
                    }
                } else {
                    if (modelAccuracyBadge) {
                        modelAccuracyBadge.innerText = "0.0%";
                        modelAccuracyBadge.style.color = "var(--accent-danger)";
                    }
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
            if (modelAccuracyBadge) modelAccuracyBadge.innerText = "Error";
        }
    }

    // Populate the list of teams on the Roster tab
    function populateTeamHealthTable(teams) {
        const tbody = document.getElementById('team-health-list');
        if (!tbody) return;
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
        const placeholder = document.getElementById('squad-editor-placeholder');
        if (placeholder) placeholder.classList.add('hidden');
        const contentDiv = document.getElementById('squad-editor-content');
        if (contentDiv) contentDiv.classList.remove('hidden');
        
        const squadNameEl = document.getElementById('active-squad-name');
        if (squadNameEl) squadNameEl.innerHTML = `Managing <strong>${teamName}</strong> player profiles`;
        
        const listBody = document.getElementById('squad-players-list');
        if (!listBody) return;
        listBody.innerHTML = '<tr><td colspan="7" class="text-muted" style="text-align: center; padding: 40px;"><i class="fa-solid fa-spinner fa-spin"></i> Retrieving player details...</td></tr>';
        
        try {
            const res = await fetch(`/api/squad/${teamName}`);
            const data = await res.json();
            
            if (data.success) {
                state.activeSquadPlayers = data.players;
                const playerCountEl = document.getElementById('player-count');
                if (playerCountEl) playerCountEl.innerText = `${data.players.length} Players Listed`;
                
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
                    
                    if (toggleInput) {
                        toggleInput.addEventListener('change', (e) => {
                            const checked = e.target.checked;
                            if (dayInjInput) dayInjInput.disabled = !checked;
                            if (gamesOutInput) gamesOutInput.disabled = !checked;
                            if (expRetInput) expRetInput.disabled = !checked;
                            
                            if (checked) {
                                // Populate logical defaults
                                const todayStr = getTodayFormattedString();
                                if (dayInjInput) dayInjInput.value = todayStr;
                                if (gamesOutInput) gamesOutInput.value = 1;
                                if (expRetInput) expRetInput.value = "Unknown";
                            } else {
                                if (dayInjInput) dayInjInput.value = '';
                                if (gamesOutInput) gamesOutInput.value = 0;
                                if (expRetInput) expRetInput.value = '';
                            }
                        });
                    }
                    
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
    const saveSquadBtn = document.getElementById('save-squad-btn');
    if (saveSquadBtn) {
        saveSquadBtn.addEventListener('click', async () => {
            if (!state.activeSquadTeam) return;
            
            const origHtml = saveSquadBtn.innerHTML;
            saveSquadBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving changes...';
            saveSquadBtn.disabled = true;
            
            const updatedPlayers = [];
            
            // Read input rows
            const toggles = document.querySelectorAll('.injury-toggle');
            toggles.forEach(tog => {
                const playerName = tog.dataset.player;
                const parentRow = tog.closest('tr');
                if (parentRow) {
                    const isInjured = tog.checked ? 1 : 0;
                    const dayInjuredInput = parentRow.querySelector('.player-day-injured');
                    const gamesOutInput = parentRow.querySelector('.player-games-out');
                    const expectedReturnInput = parentRow.querySelector('.player-expected-return');
                    
                    const dayInjured = dayInjuredInput ? dayInjuredInput.value.trim() : '';
                    const gamesOut = gamesOutInput ? (parseInt(gamesOutInput.value) || 0) : 0;
                    const expectedReturn = expectedReturnInput ? expectedReturnInput.value.trim() : '';
                    
                    updatedPlayers.push({
                        Player: playerName,
                        Injuries: isInjured,
                        "Day Injured": dayInjured,
                        "Missed Games": gamesOut,
                        "Expected Return": expectedReturn
                    });
                }
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
                saveSquadBtn.innerHTML = origHtml;
                saveSquadBtn.disabled = false;
            }
        });
    }

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
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const predictBtn = document.getElementById('predict-btn');
            const origHtml = predictBtn ? predictBtn.innerHTML : '';
            if (predictBtn) {
                predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Matchup...';
                predictBtn.disabled = true;
            }

            const payload = {
                home_team: homeTeamSelect ? homeTeamSelect.value : '',
                away_team: awayTeamSelect ? awayTeamSelect.value : '',
                home_rest_days: homeRestDays ? parseInt(homeRestDays.value) : 6,
                away_rest_days: awayRestDays ? parseInt(awayRestDays.value) : 6,
                b365h: document.getElementById('b365h') ? parseFloat(document.getElementById('b365h').value) : 2.10,
                b365d: document.getElementById('b365d') ? parseFloat(document.getElementById('b365d').value) : 3.40,
                b365a: document.getElementById('b365a') ? parseFloat(document.getElementById('b365a').value) : 3.20,
                referee: refereeSelect ? refereeSelect.value : 'Unknown'
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
                if (predictBtn) {
                    predictBtn.innerHTML = origHtml;
                    predictBtn.disabled = false;
                }
            }
        });
    }

    // Custom Canvas Drawing Helper for Polymarket-style Price Chart
    function drawPriceChartCanvas(finalProbability, outcomeName, isInitial = false) {
        const canvas = document.getElementById('price-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Determine colors based on outcomeName
        let strokeColor = '#10B981'; // Green for Home Win
        let fillColorStart = 'rgba(16, 185, 129, 0.2)';
        let fillColorEnd = 'rgba(16, 185, 129, 0.0)';

        if (outcomeName === "Away Win") {
            strokeColor = '#6366F1'; // Indigo/Blue for Away Win
            fillColorStart = 'rgba(99, 102, 241, 0.2)';
        } else if (outcomeName === "Draw") {
            strokeColor = '#F59E0B'; // Gold/Yellow for Draw
            fillColorStart = 'rgba(245, 158, 11, 0.2)';
        }

        if (isInitial) {
            strokeColor = 'rgba(148, 163, 184, 0.4)'; // Gray for initial
            fillColorStart = 'rgba(148, 163, 184, 0.05)';
        }

        // Handle high DPI screens
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const width = rect.width || canvas.width || 600;
        const height = rect.height || canvas.height || 200;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, width, height);

        // Generate points (random walk)
        const N = 50; // number of steps
        const points = [];
        
        if (isInitial) {
            // Flat line at 33% or 50%
            const val = 0.33;
            for (let i = 0; i <= N; i++) {
                points.push(val);
            }
        } else {
            // Generate a random walk ending at finalProbability
            const startVal = 0.5 + (Math.random() - 0.5) * 0.2;
            for (let i = 0; i <= N; i++) {
                const t = i / N;
                // Convergence towards final probability + random noise that decays to 0
                const noise = (1 - t) * (Math.random() - 0.5) * 0.15;
                let val = startVal + t * (finalProbability - startVal) + noise;
                // Clamp values to stay visible on chart
                val = Math.max(0.05, Math.min(0.95, val));
                points.push(val);
            }
            // Ensure the last point is exactly the final probability
            points[N] = finalProbability;
        }

        // Coordinates mapping
        const xStep = width / N;
        const getX = (i) => i * xStep;
        // 1.0 probability is at the top (y=15), 0.0 is near the bottom (y=height-15)
        const padding = 15;
        const getY = (val) => padding + (1 - val) * (height - 2 * padding);

        // Draw gradient area under the line
        ctx.beginPath();
        ctx.moveTo(getX(0), height);
        for (let i = 0; i <= N; i++) {
            ctx.lineTo(getX(i), getY(points[i]));
        }
        ctx.lineTo(getX(N), height);
        ctx.closePath();

        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, fillColorStart);
        gradient.addColorStop(1, fillColorEnd);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw smooth line
        ctx.beginPath();
        ctx.moveTo(getX(0), getY(points[0]));
        for (let i = 1; i <= N; i++) {
            ctx.lineTo(getX(i), getY(points[i]));
        }
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();

        // Draw final dot
        if (!isInitial) {
            const lastX = getX(N);
            const lastY = getY(points[N]);

            // Outer glow
            ctx.beginPath();
            ctx.arc(lastX, lastY, 6, 0, 2 * Math.PI);
            ctx.fillStyle = strokeColor;
            ctx.shadowBlur = 8;
            ctx.shadowColor = strokeColor;
            ctx.fill();

            // Reset shadow
            ctx.shadowBlur = 0;

            // Inner white dot
            ctx.beginPath();
            ctx.arc(lastX, lastY, 2, 0, 2 * Math.PI);
            ctx.fillStyle = '#FFFFFF';
            ctx.fill();

            // Draw current probability label (e.g. "65%") near the final dot
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 12px JetBrains Mono, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(`${(finalProbability * 100).toFixed(0)}%`, lastX - 10, lastY - 5);
        }
    }

    // Display prediction outcome values on screen and render Polymarket price chart
    function renderPredictionResults(res) {
        // Hide placeholder, show actual results panel
        if (predictionResultContainer) {
            const placeholder = predictionResultContainer.querySelector('.result-placeholder');
            if (placeholder) placeholder.classList.add('hidden');
            
            const actualContainer = predictionResultContainer.querySelector('.actual-results');
            if (actualContainer) actualContainer.classList.remove('hidden');
        }

        // Set outcome title
        const outcomeBadge = document.getElementById('res-prediction-outcome');
        if (outcomeBadge) {
            outcomeBadge.innerText = res.prediction.toUpperCase();
            
            // Color code outcome badge background
            if (res.prediction === "Home Win") {
                outcomeBadge.style.color = "var(--color-home-win)";
            } else if (res.prediction === "Away Win") {
                outcomeBadge.style.color = "var(--color-away-win)";
            } else {
                outcomeBadge.style.color = "var(--color-draw)";
            }
        }

        // Set probability bar widths and texts
        const homePct = (res.probabilities.home * 100).toFixed(1);
        const drawPct = (res.probabilities.draw * 100).toFixed(1);
        const awayPct = (res.probabilities.away * 100).toFixed(1);

        const homeBar = document.getElementById('prob-home-bar');
        const drawBar = document.getElementById('prob-draw-bar');
        const awayBar = document.getElementById('prob-away-bar');

        // Apply width and text content
        if (homeBar) homeBar.style.width = `${homePct}%`;
        const probHomeVal = document.getElementById('prob-home-val');
        if (probHomeVal) probHomeVal.innerText = `${homePct}%`;
        
        if (drawBar) drawBar.style.width = `${drawPct}%`;
        const probDrawVal = document.getElementById('prob-draw-val');
        if (probDrawVal) probDrawVal.innerText = `${drawPct}%`;
        
        if (awayBar) awayBar.style.width = `${awayPct}%`;
        const probAwayVal = document.getElementById('prob-away-val');
        if (probAwayVal) probAwayVal.innerText = `${awayPct}%`;

        // Update Polymarket outcome button labels and percentages
        const outcomeHomeLabel = document.getElementById('outcome-home-label');
        const outcomeAwayLabel = document.getElementById('outcome-away-label');
        if (outcomeHomeLabel) outcomeHomeLabel.innerText = `${res.home_team} Win`;
        if (outcomeAwayLabel) outcomeAwayLabel.innerText = `${res.away_team} Win`;
        
        const btnHomeProb = document.getElementById('prob-home-val-btn');
        const btnDrawProb = document.getElementById('prob-draw-val-btn');
        const btnAwayProb = document.getElementById('prob-away-val-btn');
        if (btnHomeProb) btnHomeProb.innerText = `${homePct}%`;
        if (btnDrawProb) btnDrawProb.innerText = `${drawPct}%`;
        if (btnAwayProb) btnAwayProb.innerText = `${awayPct}%`;

        // Highlight the predicted outcome button (active states/highlight colors)
        [btnHomeOutcome, btnDrawOutcome, btnAwayOutcome].forEach(btn => {
            if (btn) {
                btn.classList.remove('active');
                btn.style.backgroundColor = '';
                btn.style.borderColor = '';
            }
        });

        if (res.prediction === "Home Win" && btnHomeOutcome) {
            btnHomeOutcome.classList.add('active');
            btnHomeOutcome.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
            btnHomeOutcome.style.borderColor = 'var(--color-home-win)';
        } else if (res.prediction === "Away Win" && btnAwayOutcome) {
            btnAwayOutcome.classList.add('active');
            btnAwayOutcome.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
            btnAwayOutcome.style.borderColor = 'var(--color-away-win)';
        } else if (res.prediction === "Draw" && btnDrawOutcome) {
            btnDrawOutcome.classList.add('active');
            btnDrawOutcome.style.backgroundColor = 'rgba(245, 158, 11, 0.1)';
            btnDrawOutcome.style.borderColor = 'var(--color-draw)';
        }

        // Sync market question banner text
        const questionHome = document.getElementById('question-home-team');
        const questionAway = document.getElementById('question-away-team');
        if (questionHome) questionHome.innerText = res.home_team;
        if (questionAway) questionAway.innerText = res.away_team;

        // Render the Polymarket interactive line chart
        renderChart(res.probabilities.home, res.probabilities.draw, res.probabilities.away, res.home_team, res.away_team);

        // Set Metric values
        const metricHomeVal = document.getElementById('metric-home-value');
        if (metricHomeVal) metricHomeVal.innerText = `€${(res.home_squad_value / 1e6).toFixed(1)}M`;
        const metricAwayVal = document.getElementById('metric-away-value');
        if (metricAwayVal) metricAwayVal.innerText = `€${(res.away_squad_value / 1e6).toFixed(1)}M`;

        const metricHomeOffense = document.getElementById('metric-home-offense');
        if (metricHomeOffense) metricHomeOffense.innerText = res.home_expected_offense.toFixed(2);
        const metricAwayOffense = document.getElementById('metric-away-offense');
        if (metricAwayOffense) metricAwayOffense.innerText = res.away_expected_offense.toFixed(2);

        // Sidelined Player health summary details
        const homeTeamNameBadge = document.getElementById('home-team-name-badge');
        if (homeTeamNameBadge) homeTeamNameBadge.innerText = res.home_team;

        const homeHealthInfo = document.getElementById('home-health-info');
        if (homeHealthInfo) {
            homeHealthInfo.innerHTML = `
                <div class="health-stack">
                    <span class="health-count">${res.home_missing_key}</span>
                    <span class="health-label">key players missing</span>
                    <div class="health-sub">
                        <span class="text-danger">${res.home_missing_impact.toFixed(1)}%</span> playing impact · <span class="text-danger">${res.home_missing_goals.toFixed(1)}%</span> goals
                    </div>
                </div>
            `;
        }

        const awayTeamNameBadge = document.getElementById('away-team-name-badge');
        if (awayTeamNameBadge) awayTeamNameBadge.innerText = res.away_team;

        const awayHealthInfo = document.getElementById('away-health-info');
        if (awayHealthInfo) {
            awayHealthInfo.innerHTML = `
                <div class="health-stack">
                    <span class="health-count">${res.away_missing_key}</span>
                    <span class="health-label">key players missing</span>
                    <div class="health-sub">
                        <span class="text-danger">${res.away_missing_impact.toFixed(1)}%</span> playing impact · <span class="text-danger">${res.away_missing_goals.toFixed(1)}%</span> goals
                    </div>
                </div>
            `;
        }

        showNotification("Match Analyzed", `Prediction compiled successfully for ${res.home_team} vs ${res.away_team}.`, "success");
    }

    // Chart.js rendering helpers for Polymarket interactive feel
    function renderInitialChart() {
        drawPriceChartCanvas(0.33, 'Initial', true);
    }

    function renderChart(homeProb, drawProb, awayProb, homeTeam, awayTeam) {
        // Find the outcome with the highest probability
        let finalProb = homeProb;
        let outcomeName = "Home Win";
        
        if (awayProb > homeProb && awayProb > drawProb) {
            finalProb = awayProb;
            outcomeName = "Away Win";
        } else if (drawProb > homeProb && drawProb > awayProb) {
            finalProb = drawProb;
            outcomeName = "Draw";
        }
        
        drawPriceChartCanvas(finalProb, outcomeName, false);
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
        if (pill) {
            pill.innerText = "Running";
            pill.className = "status-pill running";
        }
        
        if (terminalStdout) terminalStdout.innerText = `[Engine] Launching subprocess: ${scriptName}...\n`;
        if (consoleTargetName) consoleTargetName.innerText = scriptName;

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
                if (pill) {
                    pill.innerText = "Failed";
                    pill.className = "status-pill failed";
                }
                if (terminalStdout) terminalStdout.innerText += `[Engine Error] Failed to launch: ${data.error}\n`;
                showNotification("Launch Failed", data.error, "danger");
            }
        } catch (err) {
            console.error(err);
            if (pill) {
                pill.innerText = "Failed";
                pill.className = "status-pill failed";
            }
            if (terminalStdout) terminalStdout.innerText += `[Engine Error] Connection crashed while executing script.\n`;
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
                    if (consoleTargetName && consoleTargetName.innerText === scriptName) {
                        if (terminalStdout) {
                            terminalStdout.innerText = data.logs || "Awaiting logs...";
                            // Auto scroll console
                            const consoleWrapper = terminalStdout.closest('.console-body-wrapper');
                            if (consoleWrapper) consoleWrapper.scrollTop = consoleWrapper.scrollHeight;
                        }
                    }

                    // Check if complete
                    const pill = document.getElementById(`status-${scriptName}`);
                    if (data.status === 'completed') {
                        if (pill) {
                            pill.innerText = "Completed";
                            pill.className = "status-pill completed";
                        }
                        clearInterval(state.logPollingIntervals[scriptName]);
                        showNotification("Process Finished", `${scriptName} has completed successfully.`, "success");
                        
                        // If it is train_model.py, grab model accuracy
                        if (scriptName === 'train_model.py') {
                            extractNewAccuracy(data.logs);
                        }
                    } else if (data.status === 'failed') {
                        if (pill) {
                            pill.innerText = "Failed";
                            pill.className = "status-pill failed";
                        }
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
            if (modelAccuracyBadge) {
                modelAccuracyBadge.innerText = `${accVal}%`;
                modelAccuracyBadge.style.color = "var(--color-gold)";
                const statValDiv = modelAccuracyBadge.closest('.quick-stat');
                if (statValDiv) {
                    statValDiv.setAttribute('title', `Prediction model trained with evaluated accuracy of ${accVal}%.`);
                }
            }
            showNotification("Accuracy Restructured", `Prediction core retrained. Accuracy recorded: ${accVal}%`, "success");
        }
    }

    // Clear console terminal
    const clearConsoleBtn = document.getElementById('clear-console-btn');
    if (clearConsoleBtn) {
        clearConsoleBtn.addEventListener('click', () => {
            if (terminalStdout) terminalStdout.innerText = "Console buffer flushed. Ready for process signals.";
        });
    }

    // TOAST NOTIFICATIONS
    function showNotification(title, message, type = 'success') {
        const toast = document.getElementById('notification-toast');
        if (!toast) return;
        const icon = toast.querySelector('.info-icon');
        const titleEl = document.getElementById('notif-title');
        const msgEl = document.getElementById('notif-message');

        if (titleEl) titleEl.innerText = title;
        if (msgEl) msgEl.innerText = message;

        // Reset classes
        toast.className = "notification";
        if (icon) icon.className = "fa-solid info-icon";

        if (type === 'success') {
            toast.style.borderColor = "rgba(16, 185, 129, 0.35)";
            if (icon) icon.classList.add('fa-circle-check', 'text-success');
        } else if (type === 'danger') {
            toast.style.borderColor = "rgba(239, 68, 68, 0.35)";
            if (icon) icon.classList.add('fa-circle-exclamation', 'text-danger');
        } else {
            // Info
            toast.style.borderColor = "rgba(226, 184, 66, 0.35)";
            if (icon) icon.classList.add('fa-circle-info', 'text-gold');
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
    const notifCloseBtn = document.getElementById('notif-close-btn');
    if (notifCloseBtn) {
        notifCloseBtn.addEventListener('click', () => {
            const toast = document.getElementById('notification-toast');
            if (toast) toast.classList.add('hidden');
        });
    }

    // Initialize application
    initializeApp();
});
