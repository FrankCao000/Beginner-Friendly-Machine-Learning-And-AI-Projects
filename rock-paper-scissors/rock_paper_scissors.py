import numpy as np

def get_computer_choice(user_history):
    if len(user_history) < 3:  # If not enough data, pick randomly
        return np.random.choice(['rock', 'paper', 'scissors'])
    
    # Convert user history to a NumPy array for analysis
    history_array = np.array(user_history)

    # Find the most common choice using NumPy
    unique, counts = np.unique(history_array, return_counts=True)
    predicted_user_choice = unique[np.argmax(counts)]  # Choice with highest frequency

    # Counter the predicted user choice
    if predicted_user_choice == 'rock':
        return 'paper'
    elif predicted_user_choice == 'paper':
        return 'scissors'
    else:
        return 'rock'

def get_user_choice():
    user_input = input("Enter your choice (rock, paper, scissors): ").lower()
    while user_input not in ['rock', 'paper', 'scissors']:
        user_input = input("Invalid choice. Try again (rock, paper, scissors): ").lower()
    return user_input

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "tie"
    elif (user_choice == 'rock' and computer_choice == 'scissors') or \
         (user_choice == 'paper' and computer_choice == 'rock') or \
         (user_choice == 'scissors' and computer_choice == 'paper'):
        return "user"
    else:
        return "computer"

def play_game():
    print("Welcome to Rock, Paper, Scissors!")
    user_score = 0
    computer_score = 0
    user_history = []  # Track user choices
    rounds = int(input("How many rounds do you want to play? "))

    for round_number in range(1, rounds + 1):
        print(f"\nRound {round_number}:")
        computer_choice = get_computer_choice(user_history)
        user_choice = get_user_choice()
        user_history.append(user_choice)
        print(f"Computer chose: {computer_choice}")
        winner = determine_winner(user_choice, computer_choice)

        if winner == "user":
            print("You win this round!")
            user_score += 1
        elif winner == "computer":
            print("Computer wins this round!")
            computer_score += 1
        else:
            print("This round is a tie!")

        print(f"Score: You {user_score} - {computer_score} Computer")

    print("\nGame Over!")
    if user_score > computer_score:
        print("Congratulations! You won the game!")
    elif user_score < computer_score:
        print("Better luck next time! The computer won!")
    else:
        print("It's a tie overall!")

if __name__ == "__main__":
    play_game()
