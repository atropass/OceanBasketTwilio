-- Создание таблицы пользователей
CREATE TABLE Users (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(255),
    ContactDetails VARCHAR(255),
    Address VARCHAR(255)
);

-- Создание таблицы жалоб
CREATE TABLE Complaints (
    ComplaintID SERIAL PRIMARY KEY,
    UserID INT,
    ComplaintType VARCHAR(255),
    ComplaintText TEXT,
    Date TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- Создание таблицы отзывов
CREATE TABLE Feedback (
    FeedbackID SERIAL PRIMARY KEY,
    UserID INT,
    FeedbackType VARCHAR(255),
    FeedbackText TEXT,
    Date TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- Создание таблицы предложений
CREATE TABLE Suggestions (
    SuggestionID SERIAL PRIMARY KEY,
    UserID INT,
    SuggestionType VARCHAR(255),
    SuggestionText TEXT,
    Date TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- Создание таблицы заказов
CREATE TABLE Orders (
    OrderID SERIAL PRIMARY KEY,
    UserID INT,
    OrderDetails TEXT,
    Address VARCHAR(255),
    OrderDate TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);
