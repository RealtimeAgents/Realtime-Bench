document.addEventListener('DOMContentLoaded', () => {
    // Game state
    const board = Array(8).fill().map(() => Array(8).fill(null));
    let turn = 'white';
    let selectedPiece = null;
    let validMoves = [];
    let moveHistory = [];
    let enPassantTarget = null;
    let castlingRights = {
        white: { kingSide: true, queenSide: true },
        black: { kingSide: true, queenSide: true }
    };
    let checkState = { white: false, black: false };
    let gameOver = false;
    
    // Game settings
    let gameMode = null; // 'local' or 'bot'
    let playerSide = 'white';
    let botSide = 'black';
    let botDifficulty = 'medium'; // 'easy', 'medium', 'hard'
    let gameTime = 600; // 10 minutes in seconds
    let timerInterval = null;
    let whiteTime = gameTime;
    let blackTime = gameTime;
    
    // DOM elements
    const chessboard = document.getElementById('chessboard');
    const historyList = document.getElementById('history-list');
    const gameStatus = document.getElementById('game-status');
    const whiteTimeDisplay = document.getElementById('white-time');
    const blackTimeDisplay = document.getElementById('black-time');
    const whitePlayerInfo = document.querySelector('.white-player');
    const blackPlayerInfo = document.querySelector('.black-player');
    
    // Initialize the board
    function initializeBoard() {
        // Clear the board
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                board[row][col] = null;
            }
        }
        
        // Set up pawns
        for (let i = 0; i < 8; i++) {
            board[1][i] = { type: 'pawn', color: 'black', hasMoved: false };
            board[6][i] = { type: 'pawn', color: 'white', hasMoved: false };
        }
        
        // Set up rooks
        board[0][0] = { type: 'rook', color: 'black', hasMoved: false };
        board[0][7] = { type: 'rook', color: 'black', hasMoved: false };
        board[7][0] = { type: 'rook', color: 'white', hasMoved: false };
        board[7][7] = { type: 'rook', color: 'white', hasMoved: false };
        
        // Set up knights
        board[0][1] = { type: 'knight', color: 'black' };
        board[0][6] = { type: 'knight', color: 'black' };
        board[7][1] = { type: 'knight', color: 'white' };
        board[7][6] = { type: 'knight', color: 'white' };
        
        // Set up bishops
        board[0][2] = { type: 'bishop', color: 'black' };
        board[0][5] = { type: 'bishop', color: 'black' };
        board[7][2] = { type: 'bishop', color: 'white' };
        board[7][5] = { type: 'bishop', color: 'white' };
        
        // Set up queens
        board[0][3] = { type: 'queen', color: 'black' };
        board[7][3] = { type: 'queen', color: 'white' };
        
        // Set up kings
        board[0][4] = { type: 'king', color: 'black', hasMoved: false };
        board[7][4] = { type: 'king', color: 'white', hasMoved: false };
        
        // Reset game state
        turn = 'white';
        selectedPiece = null;
        validMoves = [];
        moveHistory = [];
        enPassantTarget = null;
        castlingRights = {
            white: { kingSide: true, queenSide: true },
            black: { kingSide: true, queenSide: true }
        };
        checkState = { white: false, black: false };
        gameOver = false;
        
        // Reset timer if enabled
        if (gameTime > 0) {
            whiteTime = gameTime;
            blackTime = gameTime;
            updateTimerDisplay();
        }
        
        // Update UI
        updateGameStatus();
        renderBoard();
    }
    
    // Render the board
    function renderBoard() {
        chessboard.innerHTML = '';
        
        // Determine board orientation based on player side
        const rows = playerSide === 'white' ? 
            [0, 1, 2, 3, 4, 5, 6, 7] : 
            [7, 6, 5, 4, 3, 2, 1, 0];
        const cols = playerSide === 'white' ? 
            [0, 1, 2, 3, 4, 5, 6, 7] : 
            [7, 6, 5, 4, 3, 2, 1, 0];
        
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const row = rows[i];
                const col = cols[j];
                const square = document.createElement('div');
                square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.row = row;
                square.dataset.col = col;
                
                // Highlight valid moves
                if (validMoves.some(move => move.row === row && move.col === col)) {
                    square.classList.add('highlight');
                }
                
                // Highlight selected piece
                if (selectedPiece && selectedPiece.row === row && selectedPiece.col === col) {
                    square.classList.add('selected');
                }
                
                // Add piece if exists
                const piece = board[row][col];
                if (piece) {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = 'piece';
                    pieceElement.style.backgroundImage = `url('pieces/${piece.color}_${piece.type}.png')`;
                    pieceElement.dataset.row = row;
                    pieceElement.dataset.col = col;
                    square.appendChild(pieceElement);
                    
                    // Highlight king in check
                    if (piece.type === 'king' && checkState[piece.color]) {
                        square.classList.add('check');
                    }
                }
                
                square.addEventListener('click', () => handleSquareClick(row, col));
                chessboard.appendChild(square);
            }
        }
    }
    
    // Handle square click
    function handleSquareClick(row, col) {
        if (gameOver) return;
        
        // Don't allow player to move bot's pieces
        if (gameMode === 'bot' && turn !== playerSide) return;
        
        const piece = board[row][col];
        
        // If a piece of the current turn's color is clicked, select it
        if (piece && piece.color === turn) {
            selectedPiece = { row, col };
            validMoves = getValidMoves(row, col);
            renderBoard();
            return;
        }
        
        // If a square is clicked with a piece selected, try to move
        if (selectedPiece) {
            const isValidMove = validMoves.some(move => move.row === row && move.col === col);
            
            if (isValidMove) {
                makeMove(selectedPiece.row, selectedPiece.col, row, col);
                
                // If playing against bot and it's bot's turn, make bot move
                if (gameMode === 'bot' && turn === botSide && !gameOver) {
                    setTimeout(makeBotMove, 500);
                }
            } else {
                // If clicking on another piece of the same color, select that instead
                if (piece && piece.color === turn) {
                    selectedPiece = { row, col };
                    validMoves = getValidMoves(row, col);
                    renderBoard();
                } else {
                    // Otherwise, deselect
                    selectedPiece = null;
                    validMoves = [];
                    renderBoard();
                }
            }
        }
    }
    
    // Make a move
    function makeMove(fromRow, fromCol, toRow, toCol) {
        const piece = board[fromRow][fromCol];
        const moveInfo = {
            from: { row: fromRow, col: fromCol },
            to: { row: toRow, col: toCol },
            piece: { ...piece },
            captured: board[toRow][toCol] ? { ...board[toRow][toCol] } : null,
            enPassantTargetBefore: enPassantTarget,
            castlingRightsBefore: JSON.parse(JSON.stringify(castlingRights)),
            enPassant: false,
            castling: false,
            promotion: false,
            check: false
        };
        
        // Handle en passant capture
        if (piece.type === 'pawn' && toCol !== fromCol && !board[toRow][toCol]) {
            moveInfo.enPassant = true;
            moveInfo.captured = { ...board[fromRow][toCol] };
            board[fromRow][toCol] = null;
        }
        
        // Handle castling
        if (piece.type === 'king' && Math.abs(toCol - fromCol) === 2) {
            moveInfo.castling = true;
            const rookCol = toCol > fromCol ? 7 : 0;
            const newRookCol = toCol > fromCol ? 5 : 3;
            board[toRow][newRookCol] = board[toRow][rookCol];
            board[toRow][rookCol] = null;
            board[toRow][newRookCol].hasMoved = true;
        }
        
        // Move the piece
        board[toRow][toCol] = { ...piece };
        board[fromRow][fromCol] = null;
        board[toRow][toCol].hasMoved = true;
        
        // Handle pawn promotion
        if (piece.type === 'pawn' && (toRow === 0 || toRow === 7)) {
            moveInfo.promotion = true;
            // Always promote to queen for simplicity
            board[toRow][toCol].type = 'queen';
        }
        
        // Update en passant target
        enPassantTarget = null;
        if (piece.type === 'pawn' && Math.abs(toRow - fromRow) === 2) {
            enPassantTarget = { row: fromRow + (toRow - fromRow) / 2, col: fromCol };
        }
        
        // Update castling rights
        if (piece.type === 'king') {
            castlingRights[piece.color].kingSide = false;
            castlingRights[piece.color].queenSide = false;
        }
        if (piece.type === 'rook') {
            if (fromCol === 0) castlingRights[piece.color].queenSide = false;
            if (fromCol === 7) castlingRights[piece.color].kingSide = false;
        }
        
        // Check for check
        const opponentColor = piece.color === 'white' ? 'black' : 'white';
        const isCheck = isKingInCheck(opponentColor);
        moveInfo.check = isCheck;
        checkState[opponentColor] = isCheck;
        
        // Switch turns
        turn = opponentColor;
        selectedPiece = null;
        validMoves = [];
        
        moveHistory.push(moveInfo);
        updateMoveHistory();
        updateGameStatus();
        renderBoard();
        
        // Check for game over conditions
        if (isCheck) {
            if (isCheckmate(opponentColor)) {
                gameOver = true;
                gameStatus.textContent = `Checkmate! ${piece.color === 'white' ? 'White' : 'Black'} wins!`;
                stopTimer();
            }
        } else if (isStalemate(opponentColor)) {
            gameOver = true;
            gameStatus.textContent = 'Stalemate! Game drawn.';
            stopTimer();
        }
        
        // Start timer for the next player if enabled
        if (gameTime > 0 && !gameOver) {
            startTimer();
        }
    }
    
    // Get valid moves for a piece
    function getValidMoves(row, col, skipTurnCheck = false) {
        const piece = board[row][col];
        if (!piece) return [];
        if (!skipTurnCheck && piece.color !== turn) return [];
        
        const moves = [];
        
        switch (piece.type) {
            case 'pawn':
                const direction = piece.color === 'white' ? -1 : 1;
                const startRow = piece.color === 'white' ? 6 : 1;
                
                // Move forward
                if (isInBounds(row + direction, col) && !board[row + direction][col]) {
                    moves.push({ row: row + direction, col });
                    
                    // Double move from starting position
                    if (row === startRow && !board[row + 2 * direction][col] && !board[row + direction][col]) {
                        moves.push({ row: row + 2 * direction, col });
                    }
                }
                
                // Capture diagonally
                for (const dc of [-1, 1]) {
                    const newCol = col + dc;
                    if (isInBounds(row + direction, newCol)) {
                        // Normal capture
                        if (board[row + direction][newCol] && board[row + direction][newCol].color !== piece.color) {
                            moves.push({ row: row + direction, col: newCol });
                        }
                        // En passant
                        if (enPassantTarget && enPassantTarget.row === row && enPassantTarget.col === newCol) {
                            moves.push({ row: row + direction, col: newCol });
                        }
                    }
                }
                break;
                
            case 'rook':
                addStraightMoves(row, col, piece.color, moves);
                break;
                
            case 'knight':
                const knightMoves = [
                    { dr: 2, dc: 1 }, { dr: 2, dc: -1 },
                    { dr: -2, dc: 1 }, { dr: -2, dc: -1 },
                    { dr: 1, dc: 2 }, { dr: 1, dc: -2 },
                    { dr: -1, dc: 2 }, { dr: -1, dc: -2 }
                ];
                
                for (const move of knightMoves) {
                    const newRow = row + move.dr;
                    const newCol = col + move.dc;
                    if (isInBounds(newRow, newCol) && 
                        (!board[newRow][newCol] || board[newRow][newCol].color !== piece.color)) {
                        moves.push({ row: newRow, col: newCol });
                    }
                }
                break;
                
            case 'bishop':
                addDiagonalMoves(row, col, piece.color, moves);
                break;
                
            case 'queen':
                addStraightMoves(row, col, piece.color, moves);
                addDiagonalMoves(row, col, piece.color, moves);
                break;
                
            case 'king':
                for (let dr = -1; dr <= 1; dr++) {
                    for (let dc = -1; dc <= 1; dc++) {
                        if (dr === 0 && dc === 0) continue;
                        const newRow = row + dr;
                        const newCol = col + dc;
                        if (isInBounds(newRow, newCol) && 
                            (!board[newRow][newCol] || board[newRow][newCol].color !== piece.color)) {
                            moves.push({ row: newRow, col: newCol });
                        }
                    }
                }
                
                // Castling
                if (!piece.hasMoved && !checkState[piece.color]) {
                    const color = piece.color;
                    const backRow = color === 'white' ? 7 : 0;
                    
                    // King-side castling
                    if (castlingRights[color].kingSide && 
                        !board[backRow][5] && !board[backRow][6] &&
                        board[backRow][7] && board[backRow][7].type === 'rook' && !board[backRow][7].hasMoved) {
                        // Check if squares are under attack
                        if (!isSquareUnderAttack(backRow, 5, color) && !isSquareUnderAttack(backRow, 6, color)) {
                            moves.push({ row: backRow, col: 6, castling: 'kingSide' });
                        }
                    }
                    
                    // Queen-side castling
                    if (castlingRights[color].queenSide && 
                        !board[backRow][3] && !board[backRow][2] && !board[backRow][1] &&
                        board[backRow][0] && board[backRow][0].type === 'rook' && !board[backRow][0].hasMoved) {
                        // Check if squares are under attack
                        if (!isSquareUnderAttack(backRow, 3, color) && !isSquareUnderAttack(backRow, 2, color)) {
                            moves.push({ row: backRow, col: 2, castling: 'queenSide' });
                        }
                    }
                }
                break;
        }
        
        // Filter out moves that would leave king in check
        if (!skipTurnCheck) {
            return moves.filter(move => {
                // Simulate the move
                const originalBoard = JSON.parse(JSON.stringify(board));
                const originalPiece = board[move.row][move.col];
                
                board[move.row][move.col] = board[row][col];
                board[row][col] = null;
                
                // Handle castling rook move
                if (move.castling) {
                    const rookCol = move.col === 6 ? 7 : 0;
                    const newRookCol = move.col === 6 ? 5 : 3;
                    board[move.row][newRookCol] = board[move.row][rookCol];
                    board[move.row][rookCol] = null;
                }
                
                // Handle en passant capture
                if (board[move.row][move.col].type === 'pawn' && move.col !== col && !originalPiece) {
                    board[row][move.col] = null;
                }
                
                const isCheck = isKingInCheck(turn);
                
                // Restore board
                board[row][col] = originalBoard[row][col];
                board[move.row][move.col] = originalBoard[move.row][move.col];
                
                // Restore castling rook if needed
                if (move.castling) {
                    const rookCol = move.col === 6 ? 7 : 0;
                    const newRookCol = move.col === 6 ? 5 : 3;
                    board[move.row][rookCol] = originalBoard[move.row][rookCol];
                    board[move.row][newRookCol] = originalBoard[move.row][newRookCol];
                }
                
                return !isCheck;
            });
        }
        
        return moves;
    }
    
    // Helper function to add straight moves
    function addStraightMoves(row, col, color, moves) {
        const directions = [
            { dr: 1, dc: 0 }, { dr: -1, dc: 0 },
            { dr: 0, dc: 1 }, { dr: 0, dc: -1 }
        ];
        
        for (const dir of directions) {
            for (let i = 1; i < 8; i++) {
                const newRow = row + i * dir.dr;
                const newCol = col + i * dir.dc;
                
                if (!isInBounds(newRow, newCol)) break;
                
                if (!board[newRow][newCol]) {
                    moves.push({ row: newRow, col: newCol });
                } else {
                    if (board[newRow][newCol].color !== color) {
                        moves.push({ row: newRow, col: newCol });
                    }
                    break;
                }
            }
        }
    }
    
    // Helper function to add diagonal moves
    function addDiagonalMoves(row, col, color, moves) {
        const directions = [
            { dr: 1, dc: 1 }, { dr: 1, dc: -1 },
            { dr: -1, dc: 1 }, { dr: -1, dc: -1 }
        ];
        
        for (const dir of directions) {
            for (let i = 1; i < 8; i++) {
                const newRow = row + i * dir.dr;
                const newCol = col + i * dir.dc;
                
                if (!isInBounds(newRow, newCol)) break;
                
                if (!board[newRow][newCol]) {
                    moves.push({ row: newRow, col: newCol });
                } else {
                    if (board[newRow][newCol].color !== color) {
                        moves.push({ row: newRow, col: newCol });
                    }
                    break;
                }
            }
        }
    }
    
    // Check if coordinates are within bounds
    function isInBounds(row, col) {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }
    
    // Check if square is under attack
    function isSquareUnderAttack(row, col, color) {
        const opponentColor = color === 'white' ? 'black' : 'white';
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const piece = board[r][c];
                if (piece && piece.color === opponentColor) {
                    const moves = getValidMoves(r, c, true);
                    if (moves.some(m => m.row === row && m.col === col)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    // Check if king is in check
    function isKingInCheck(color) {
        // Find king position
        let kingPos = null;
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (piece && piece.type === 'king' && piece.color === color) {
                    kingPos = { row, col };
                    break;
                }
            }
            if (kingPos) break;
        }
        
        if (!kingPos) return false;
        
        // Check if any opponent piece can attack the king
        return isSquareUnderAttack(kingPos.row, kingPos.col, color);
    }
    
    // Check for checkmate
    function isCheckmate(color) {
        // King must be in check
        if (!checkState[color]) return false;
        
        // Check if any move can get out of check
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (piece && piece.color === color) {
                    const moves = getValidMoves(row, col);
                    if (moves.length > 0) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }
    
    // Check for stalemate
    function isStalemate(color) {
        // King must not be in check
        if (checkState[color]) return false;
        
        // Check if any legal move is available
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (piece && piece.color === color) {
                    const moves = getValidMoves(row, col);
                    if (moves.length > 0) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }
    
    // Update move history display
    function updateMoveHistory() {
        historyList.innerHTML = '';
        
        for (let i = 0; i < moveHistory.length; i += 2) {
            const moveContainer = document.createElement('div');
            moveContainer.className = 'move-entry white';
            moveContainer.textContent = `${i/2 + 1}. ${formatMove(moveHistory[i])}`;
            historyList.appendChild(moveContainer);
            
            if (i + 1 < moveHistory.length) {
                const blackMoveContainer = document.createElement('div');
                blackMoveContainer.className = 'move-entry black';
                blackMoveContainer.textContent = formatMove(moveHistory[i + 1]);
                historyList.appendChild(blackMoveContainer);
            }
        }
        
        // Scroll to bottom
        historyList.scrollTop = historyList.scrollHeight;
    }
    
    // Format a move for display
    function formatMove(move) {
        const piece = move.piece.type === 'pawn' ? '' : move.piece.type.charAt(0).toUpperCase();
        const capture = move.captured ? 'x' : '';
        const from = String.fromCharCode(97 + move.from.col) + (8 - move.from.row);
        const to = String.fromCharCode(97 + move.to.col) + (8 - move.to.row);
        
        // Special move notations
        if (move.castling) {
            return move.to.col === 6 ? 'O-O' : 'O-O-O';
        }
        if (move.enPassant) {
            return `${from[0]}x${to} e.p.`;
        }
        if (move.promotion) {
            return `${to}=Q`;
        }
        if (move.check) {
            return `${piece}${from}${capture}${to}+`;
        }
        
        return `${piece}${from}${capture}${to}`;
    }
    
    // Update game status display
    function updateGameStatus() {
        if (gameOver) return;
        
        let status = `${turn === 'white' ? 'White' : 'Black'}'s turn`;
        if (checkState[turn === 'white' ? 'black' : 'white']) {
            status += ' (Check!)';
        }
        gameStatus.textContent = status;
        
        // Update active player highlight
        if (turn === 'white') {
            whitePlayerInfo.classList.add('active-player');
            blackPlayerInfo.classList.remove('active-player');
        } else {
            whitePlayerInfo.classList.remove('active-player');
            blackPlayerInfo.classList.add('active-player');
        }
    }
    
    // Make a move for the bot
    function makeBotMove() {
        if (gameOver) return;
        
        // Find all possible moves for bot pieces
        const allMoves = [];
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (piece && piece.color === botSide) {
                    const moves = getValidMoves(row, col);
                    for (const move of moves) {
                        allMoves.push({
                            from: { row, col },
                            to: move,
                            piece: piece
                        });
                    }
                }
            }
        }
        
        if (allMoves.length === 0) {
            // Check for checkmate or stalemate
            if (checkState[botSide]) {
                gameOver = true;
                gameStatus.textContent = `Checkmate! ${botSide === 'white' ? 'Black' : 'White'} wins!`;
                stopTimer();
            } else {
                gameOver = true;
                gameStatus.textContent = 'Stalemate! Game drawn.';
                stopTimer();
            }
            return;
        }
        
        let chosenMove;
        
        switch (botDifficulty) {
            case 'easy':
                // Random moves
                chosenMove = allMoves[Math.floor(Math.random() * allMoves.length)];
                break;
                
            case 'medium':
                // 1-ply: prioritize captures and checks
                chosenMove = findBestMove(allMoves, 1);
                break;
                
            case 'hard':
                // 2-3 ply: more advanced evaluation
                chosenMove = findBestMove(allMoves, 3);
                break;
                
            default:
                chosenMove = allMoves[Math.floor(Math.random() * allMoves.length)];
        }
        
        makeMove(chosenMove.from.row, chosenMove.from.col, chosenMove.to.row, chosenMove.to.col);
    }
    
    // Find the best move using minimax algorithm
    function findBestMove(moves, depth) {
        // Simple evaluation function
        function evaluateBoard() {
            let score = 0;
            const pieceValues = {
                pawn: 1,
                knight: 3,
                bishop: 3,
                rook: 5,
                queen: 9,
                king: 0 // King value not considered in evaluation
            };
            
            for (let row = 0; row < 8; row++) {
                for (let col = 0; col < 8; col++) {
                    const piece = board[row][col];
                    if (piece) {
                        const value = pieceValues[piece.type];
                        score += piece.color === botSide ? value : -value;
                    }
                }
            }
            
            return score;
        }
        
        // If depth is 0 or no moves, return evaluation
        if (depth === 0 || moves.length === 0) {
            return moves[Math.floor(Math.random() * moves.length)];
        }
        
        // For medium difficulty (1-ply), just look for captures and checks
        if (depth === 1) {
            // Find checking moves
            const checkingMoves = moves.filter(move => {
                // Simulate move to see if it puts opponent in check
                const originalBoard = JSON.parse(JSON.stringify(board));
                const originalPiece = board[move.to.row][move.to.col];
                
                board[move.to.row][move.to.col] = board[move.from.row][move.from.col];
                board[move.from.row][move.from.col] = null;
                
                const isCheck = isKingInCheck(playerSide);
                
                // Restore board
                board[move.from.row][move.from.col] = originalBoard[move.from.row][move.from.col];
                board[move.to.row][move.to.col] = originalBoard[move.to.row][move.to.col];
                
                return isCheck;
            });
            
            if (checkingMoves.length > 0) {
                // Prefer checking moves that also capture
                const checkingCaptures = checkingMoves.filter(move => 
                    board[move.to.row][move.to.col] ||
                    (board[move.from.row][move.from.col].type === 'pawn' && 
                     move.to.col !== move.from.col && !board[move.to.row][move.to.col])
                );
                
                return checkingCaptures.length > 0 ? 
                    checkingCaptures[Math.floor(Math.random() * checkingCaptures.length)] :
                    checkingMoves[Math.floor(Math.random() * checkingMoves.length)];
            }
            
            // Find capturing moves
            const capturingMoves = moves.filter(move => 
                board[move.to.row][move.to.col] || 
                (board[move.from.row][move.from.col].type === 'pawn' && 
                 move.to.col !== move.from.col && !board[move.to.row][move.to.col])
            );
            
            if (capturingMoves.length > 0) {
                return capturingMoves[Math.floor(Math.random() * capturingMoves.length)];
            }
            
            // Otherwise random move
            return moves[Math.floor(Math.random() * moves.length)];
        }
        
        // For hard difficulty (2-3 ply), use minimax with simple evaluation
        let bestMove = null;
        let bestValue = -Infinity;
        
        for (const move of moves) {
            // Make the move
            const originalBoard = JSON.parse(JSON.stringify(board));
            const originalPiece = board[move.to.row][move.to.col];
            
            board[move.to.row][move.to.col] = board[move.from.row][move.from.col];
            board[move.from.row][move.from.col] = null;
            
            // Evaluate the position
            let value = -evaluateBoard();
            
            // If we have depth left, look deeper
            if (depth > 1) {
                // Get opponent's moves
                const opponentMoves = [];
                for (let row = 0; row < 8; row++) {
                    for (let col = 0; col < 8; col++) {
                        const piece = board[row][col];
                        if (piece && piece.color === playerSide) {
                            const pieceMoves = getValidMoves(row, col);
                            for (const pmove of pieceMoves) {
                                opponentMoves.push({
                                    from: { row, col },
                                    to: pmove,
                                    piece: piece
                                });
                            }
                        }
                    }
                }
                
                // Find opponent's best response
                if (opponentMoves.length > 0) {
                    let opponentBestValue = Infinity;
                    
                    for (const oppMove of opponentMoves) {
                        // Make opponent's move
                        const oppOriginalPiece = board[oppMove.to.row][oppMove.to.col];
                        
                        board[oppMove.to.row][oppMove.to.col] = board[oppMove.from.row][oppMove.from.col];
                        board[oppMove.from.row][oppMove.from.col] = null;
                        
                        // Evaluate the position
                        const oppValue = evaluateBoard();
                        
                        if (oppValue < opponentBestValue) {
                            opponentBestValue = oppValue;
                        }
                        
                        // Undo opponent's move
                        board[oppMove.from.row][oppMove.from.col] = board[oppMove.to.row][oppMove.to.col];
                        board[oppMove.to.row][oppMove.to.col] = oppOriginalPiece;
                    }
                    
                    value = -opponentBestValue;
                }
            }
            
            // Undo the move
            board[move.from.row][move.from.col] = board[move.to.row][move.to.col];
            board[move.to.row][move.to.col] = originalPiece;
            
            // Update best move
            if (value > bestValue) {
                bestValue = value;
                bestMove = move;
            }
        }
        
        return bestMove || moves[Math.floor(Math.random() * moves.length)];
    }
    
    // Show hint
    function showHint() {
        if (gameOver || turn !== playerSide) return;
        
        // Find all possible moves for player
        const allMoves = [];
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                if (piece && piece.color === playerSide) {
                    const moves = getValidMoves(row, col);
                    for (const move of moves) {
                        allMoves.push({
                            from: { row, col },
                            to: move
                        });
                    }
                }
            }
        }
        
        if (allMoves.length === 0) return;
        
        // Find checking moves
        const checkingMoves = allMoves.filter(move => {
            // Simulate move to see if it puts opponent in check
            const originalBoard = JSON.parse(JSON.stringify(board));
            const originalPiece = board[move.to.row][move.to.col];
            
            board[move.to.row][move.to.col] = board[move.from.row][move.from.col];
            board[move.from.row][move.from.col] = null;
            
            const isCheck = isKingInCheck(botSide);
            
            // Restore board
            board[move.from.row][move.from.col] = originalBoard[move.from.row][move.from.col];
            board[move.to.row][move.to.col] = originalBoard[move.to.row][move.to.col];
            
            return isCheck;
        });
        
        // Find capturing moves
        const capturingMoves = allMoves.filter(move => 
            board[move.to.row][move.to.col] || 
            (board[move.from.row][move.from.col].type === 'pawn' && 
             move.to.col !== move.from.col && !board[move.to.row][move.to.col])
        );
        
        let hintMove;
        
        // Prefer checking moves
        if (checkingMoves.length > 0) {
            // Prefer checking captures
            const checkingCaptures = checkingMoves.filter(move => 
                board[move.to.row][move.to.col] ||
                (board[move.from.row][move.from.col].type === 'pawn' && 
                 move.to.col !== move.from.col && !board[move.to.row][move.to.col])
            );
            
            hintMove = checkingCaptures.length > 0 ? 
                checkingCaptures[Math.floor(Math.random() * checkingCaptures.length)] :
                checkingMoves[Math.floor(Math.random() * checkingMoves.length)];
        }
        // Then prefer captures
        else if (capturingMoves.length > 0) {
            hintMove = capturingMoves[Math.floor(Math.random() * capturingMoves.length)];
        }
        // Otherwise random move
        else {
            hintMove = allMoves[Math.floor(Math.random() * allMoves.length)];
        }
        
        // Highlight the suggested move
        selectedPiece = { row: hintMove.from.row, col: hintMove.from.col };
        validMoves = [hintMove.to];
        renderBoard();
        
        // Clear the hint after 2 seconds
        setTimeout(() => {
            if (selectedPiece && 
                selectedPiece.row === hintMove.from.row && 
                selectedPiece.col === hintMove.from.col) {
                selectedPiece = null;
                validMoves = [];
                renderBoard();
            }
        }, 2000);
    }
    
    // Undo the last move
    function undoMove() {
        if (moveHistory.length === 0 || gameOver) return;
        
        const lastMove = moveHistory.pop();
        
        // Restore the moved piece
        board[lastMove.from.row][lastMove.from.col] = lastMove.piece;
        
        // Restore the captured piece (if any)
        board[lastMove.to.row][lastMove.to.col] = lastMove.captured;
        
        // Restore en passant target
        enPassantTarget = lastMove.enPassantTargetBefore;
        
        // Restore castling rights
        castlingRights = lastMove.castlingRightsBefore;
        
        // Handle en passant undo
        if (lastMove.enPassant) {
            board[lastMove.from.row][lastMove.to.col] = lastMove.captured;
        }
        
        // Handle castling undo
        if (lastMove.castling) {
            const rookCol = lastMove.to.col > lastMove.from.col ? 7 : 0;
            const newRookCol = lastMove.to.col > lastMove.from.col ? 5 : 3;
            board[lastMove.to.row][rookCol] = board[lastMove.to.row][newRookCol];
            board[lastMove.to.row][newRookCol] = null;
            board[lastMove.to.row][rookCol].hasMoved = false;
        }
        
        // Handle promotion undo
        if (lastMove.promotion) {
            board[lastMove.from.row][lastMove.from.col].type = 'pawn';
        }
        
        // Restore check state
        checkState.white = false;
        checkState.black = false;
        if (moveHistory.length > 0) {
            const prevMove = moveHistory[moveHistory.length - 1];
            if (prevMove.check) {
                checkState[prevMove.piece.color === 'white' ? 'black' : 'white'] = true;
            }
        }
        
        // Switch turns back
        turn = turn === 'white' ? 'black' : 'white';
        gameOver = false;
        
        updateMoveHistory();
        updateGameStatus();
        renderBoard();
        
        // If we're in bot mode and it's now bot's turn again, undo one more move
        if (gameMode === 'bot' && turn === botSide && moveHistory.length > 0) {
            undoMove();
        }
        
        // Restart timer if enabled
        if (gameTime > 0) {
            startTimer();
        }
    }
    
    // Start the game timer
    function startTimer() {
        stopTimer();
        
        if (gameTime === 0) return;
        
        timerInterval = setInterval(() => {
            if (turn === 'white') {
                whiteTime--;
                if (whiteTime <= 0) {
                    whiteTime = 0;
                    gameOver = true;
                    gameStatus.textContent = 'Time out! Black wins!';
                    stopTimer();
                }
            } else {
                blackTime--;
                if (blackTime <= 0) {
                    blackTime = 0;
                    gameOver = true;
                    gameStatus.textContent = 'Time out! White wins!';
                    stopTimer();
                }
            }
            
            updateTimerDisplay();
        }, 1000);
    }
    
    // Stop the game timer
    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }
    
    // Update timer display
    function updateTimerDisplay() {
        if (gameTime === 0) {
            whiteTimeDisplay.textContent = '--:--';
            blackTimeDisplay.textContent = '--:--';
            return;
        }
        
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
        };
        
        whiteTimeDisplay.textContent = formatTime(whiteTime);
        blackTimeDisplay.textContent = formatTime(blackTime);
    }
    
    // Set up event listeners
    document.getElementById('local-mode').addEventListener('click', () => {
        gameMode = 'local';
        document.querySelectorAll('.setup-section').forEach(el => el.style.display = 'block');
        document.getElementById('game-status').textContent = 'Choose your settings to begin';
    });
    
    document.getElementById('bot-mode').addEventListener('click', () => {
        gameMode = 'bot';
        document.querySelectorAll('.setup-section').forEach(el => el.style.display = 'block');
        document.getElementById('game-status').textContent = 'Choose your settings to begin';
    });
    
    // Bot difficulty selection
    document.querySelectorAll('#bot-difficulty button').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('#bot-difficulty button').forEach(btn => 
                btn.classList.remove('active'));
            button.classList.add('active');
            botDifficulty = button.dataset.difficulty;
        });
    });
    
    // Side selection
    document.querySelectorAll('#side-selection button').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('#side-selection button').forEach(btn => 
                btn.classList.remove('active'));
            button.classList.add('active');
            
            let side = button.dataset.side;
            if (side === 'random') {
                side = Math.random() < 0.5 ? 'white' : 'black';
                document.querySelector(`#side-selection button[data-side="${side}"]`).classList.add('active');
            }
            
            playerSide = side;
            botSide = side === 'white' ? 'black' : 'white';
        });
    });
    
    // Timer selection
    document.querySelectorAll('#timer-selection button').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('#timer-selection button').forEach(btn => 
                btn.classList.remove('active'));
            button.classList.add('active');
            gameTime = parseInt(button.dataset.time) * 60 || 0;
        });
    });
    
    // Game controls
    document.getElementById('hint-btn').addEventListener('click', showHint);
    document.getElementById('undo-btn').addEventListener('click', undoMove);
    document.getElementById('restart-btn').addEventListener('click', () => {
        if (confirm('Are you sure you want to restart the game?')) {
            initializeBoard();
        }
    });
    
    // Initialize the game
    initializeBoard();
    
    // Start with default selections
    document.querySelector('#bot-difficulty button[data-difficulty="medium"]').classList.add('active');
    document.querySelector('#side-selection button[data-side="white"]').classList.add('active');
    document.querySelector('#timer-selection button[data-time="10"]').classList.add('active');
});