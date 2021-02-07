//package com.deep.framework.jit.ir;
//
//public class RecursionDescendParser {
//
//    private BlockLexer lexer = null;
//    private Token lookAhead = null;
//
//    public RecursionDescendParser() {
//    }
//
//    public void doParse(String filePath) {
//        lexer = new BlockLexer(filePath);
//        this.parse();
//    }
//
//    public void matchToken(TokenType type, String functionName) {
//        System.out.println("In matchToken();-----jmzhang-----" + lookAhead.getType());
//        if (lookAhead.getType() != type) {
//            parsingError(type.toString(), functionName);
//        }
//        lookAhead = lexer.nextToken();
//    }
//
//    public void parsingError(String types, String functionName) {
//        System.out.println("Parsing Error! in" + functionName);
//        System.out.println("encounter " + lookAhead.getLexeme());
//        System.out.println("at line " + lookAhead.getLine() + ",column " + lookAhead.getColumn());
//        System.out.println("while expecting " + types);
//        System.exit(1);
//    }
//
//    /**
//     * 调用开始符号对应的方法，进行语法分析
//     * 方法名：parse
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-上午10:27:14
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    public void parse() {
//        System.out.println("In parse();-----jmzhang-----");
//        lookAhead = lexer.nextToken();
//        simpleblock();
//        System.out.println("Parsing Success!");
//    }
//
//    public void simpleblock() {
//        System.out.println("In simpleblock();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.LBRACKET) {
//            matchToken(TokenType.LBRACKET, "simpleblock");
//            System.out.println("In simpleblock();-----jmzhang-----11");
//            Sequence();
//            System.out.println("In simpleblock();-----jmzhang-----12");
//            matchToken(TokenType.RBRACKET, "simpleblock");
//            System.out.println("In simpleblock();-----jmzhang-----13");
//            if (lookAhead.getType() == TokenType.LBRACKET) {
//                simpleblock();
//            }
//        } else {
//            System.out.println("In simpleblock();-----jmzhang-----2");
//            parsingError(TokenType.LBRACKET.toString(), "simpleblock");
//        }
//    }
//
//    /**
//     * sequence=assognmentStatement sequence |
//     * ifStatement sequence |
//     * whileStatement sequence |
//     * epsilon
//     * S->AS | IS |WS | ε
//     * 方法名：Sequence
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午8:54:23
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void Sequence() {
//        System.out.println("In Sequence();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.IDENTIFIER) {
//            System.out.println("In Sequence();-----jmzhang-----1");
//            assignmentStatement();
//            Sequence();
//        } else if (lookAhead.getType() == TokenType.KEY_IF) {
//            System.out.println("In Sequence();-----jmzhang-----2");
//            ifStatement();
//            Sequence();
//        } else if (lookAhead.getType() == TokenType.KEY_WHILE) {
//            System.out.println("In Sequence();-----jmzhang-----WHILE");
//            whileStatement();
//            Sequence();
//        } else if (lookAhead.getType() == TokenType.RBRACKET) {
//            //match epslon
//            System.out.println("In Sequence();-----jmzhang-----3");
//        } else {
//            System.out.println("In Sequence();-----jmzhang-----4");
//            String errorTypes = TokenType.IDENTIFIER.toString() + "," + TokenType.RBRACKET.toString();
//            parsingError(errorTypes, "sequence");
//        }
//    }
//
//    /**
//     * 方法名：whileStatement
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:31:56
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void whileStatement() {
//        if (lookAhead.getType() == TokenType.KEY_WHILE) {
//            System.out.println("In whileStatement();-----jmzhang-----");
//            matchToken(TokenType.KEY_WHILE, "whileStatement");
//            matchToken(TokenType.LPAREN, "whileStatement");
//            Boolexpression();
//            matchToken(TokenType.RPAREN, "whileStatement");
//            matchToken(TokenType.LBRACKET, "whileStatement");
//            Sequence();
//            matchToken(TokenType.RBRACKET, "whileStatement");
//        } else {
//            String errorTypes = TokenType.KEY_WHILE.toString();
//            parsingError(errorTypes, "whileStatement");
//        }
//
//    }
//
//    /**
//     * 方法名：Boolexpression
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:23:48
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void Boolexpression() {
//        System.out.println("In Boolexpression();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.BOOL_TRUE
//                || lookAhead.getType() == TokenType.BOOL_FALSE
//                || lookAhead.getType() == TokenType.LPAREN
//                || lookAhead.getType() == TokenType.IDENTIFIER
//                || lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            Boolterm();
//            Boolexpression_1();
//        } else {
//            String errorTypes = TokenType.BOOL_TRUE.toString()
//                    + "," + TokenType.BOOL_FALSE.toString()
//                    + "," + TokenType.LPAREN.toString()
//                    + "," + TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString();
//            parsingError(errorTypes, "Boolexpression");
//        }
//    }
//
//    /**
//     * 方法名：Boolexpression_1
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:22:10
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void Boolexpression_1() {
//        System.out.println("In Boolexpression_1();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.LOGICAL_OR) {
//            matchToken(TokenType.LOGICAL_OR, "Boolexpression_1");
//            Boolterm();
//            Boolexpression_1();
//        } else if (lookAhead.getType() == TokenType.RPAREN) {
//            //match epslin
//            //follow(E')={')',','}
//        } else {
//            String errorTypes = TokenType.LOGICAL_OR.toString()
//                    + "," + TokenType.RPAREN.toString();
//            parsingError(errorTypes, "Boolexpression_1");
//        }
//    }
//
//    private void Boolterm() {
//        System.out.println("In Boolterm();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.BOOL_TRUE
//                || lookAhead.getType() == TokenType.BOOL_FALSE
//                || lookAhead.getType() == TokenType.LPAREN
//                || lookAhead.getType() == TokenType.IDENTIFIER
//                || lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            Boolfactor();
//            Boolterm_1();
//        } else {
//            String errorTypes = TokenType.BOOL_TRUE.toString()
//                    + "," + TokenType.BOOL_FALSE.toString()
//                    + "," + TokenType.LPAREN.toString()
//                    + "," + TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString();
//            parsingError(errorTypes, "Boolterm");
//        }
//    }
//
//    /**
//     * 方法名：Boolterm_1
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:12:13
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void Boolterm_1() {
//        System.out.println("In Boolterm_1();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.LOGICAL_AND) {
//            matchToken(TokenType.LOGICAL_AND, "Boolterm_1");
//            Boolfactor();
//            Boolterm_1();
//        } else if (lookAhead.getType() == TokenType.LOGICAL_OR
//                || lookAhead.getType() == TokenType.RPAREN) {
//            //match e[slion
//            //follow(T')={'+','-',')',',')
//        } else {
//            String errorTypes = TokenType.LOGICAL_AND.toString()
//
//                    + "," + TokenType.LOGICAL_OR.toString()
//                    + "," + TokenType.RPAREN.toString()
//                    + "," + TokenType.SEMICOLON.toString();
//            parsingError(errorTypes, "Boolterm_1");
//        }
//    }
//
//    /**
//     * 方法名：Boolfactor
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:09:50
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void Boolfactor() {
//        System.out.println("In Boolfactor();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.BOOL_TRUE) {
//            matchToken(TokenType.BOOL_TRUE, "Boolfactor");
//        } else if (lookAhead.getType() == TokenType.BOOL_FALSE) {
//            matchToken(TokenType.BOOL_FALSE, "Boolfactor");
//        } else if (lookAhead.getType() == TokenType.LPAREN ||
//                lookAhead.getType() == TokenType.IDENTIFIER ||
//                lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            relationlExpression();
//        } else {
//            String errorTypes = TokenType.BOOL_TRUE.toString()
//                    + "," + TokenType.BOOL_FALSE.toString()
//                    + "," + TokenType.LPAREN.toString()
//                    + "," + TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString();
//            parsingError(errorTypes, "relationlExpressionOperator");
//        }
//    }
//
//    /**
//     * 方法名：relationlExpression
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:09:41
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void relationlExpression() {
//        if (lookAhead.getType() == TokenType.LPAREN ||
//                lookAhead.getType() == TokenType.IDENTIFIER ||
//                lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            System.out.println("In relationlExpression();-----jmzhang-----");
//            expression();
//            relationlExpressionOperator();
//            expression();
//        } else {
//            String errorTypes = TokenType.LPAREN.toString()
//                    + "," + TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString();
//            parsingError(errorTypes, "relationlExpression");
//        }
//    }
//
//    /**
//     * 方法名：relationlExpressionOperator
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:09:30
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void relationlExpressionOperator() {
//        System.out.println("In relationlExpressionOperator();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.LESS) {
//            matchToken(TokenType.LESS, "relationlExpressionOperator");
//        } else if (lookAhead.getType() == TokenType.GREATER) {
//            matchToken(TokenType.GREATER, "relationlExpressionOperator");
//        } else if (lookAhead.getType() == TokenType.LESS_EQUAL) {
//            matchToken(TokenType.LESS_EQUAL, "relationlExpressionOperator");
//        } else if (lookAhead.getType() == TokenType.GREATER_EQUAL) {
//            matchToken(TokenType.GREATER_EQUAL, "relationlExpressionOperator");
//        } else if (lookAhead.getType() == TokenType.NOT_EQUAL) {
//            matchToken(TokenType.NOT_EQUAL, "relationlExpressionOperator");
//        } else if (lookAhead.getType() == TokenType.EQUAL) {
//            matchToken(TokenType.EQUAL, "relationlExpressionOperator");
//        } else {
//            String errorTypes = TokenType.LESS.toString()
//                    + "," + TokenType.GREATER.toString()
//                    + "," + TokenType.LESS_EQUAL.toString()
//                    + "," + TokenType.GREATER_EQUAL.toString()
//                    + "," + TokenType.NOT_EQUAL.toString()
//                    + "," + TokenType.EQUAL.toString();
//            parsingError(errorTypes, "relationlExpressionOperator");
//        }
//
//    }
//
//    /**
//     * 方法名：ifStatement
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:29:16
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void ifStatement() {
//        System.out.println("In ifStatement();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.KEY_IF) {
//            matchToken(TokenType.KEY_IF, "ifStatement");
//            matchToken(TokenType.LPAREN, "ifStatement");
//            Boolexpression();
//            matchToken(TokenType.RPAREN, "ifStatement");
//            matchToken(TokenType.LBRACKET, "ifStatement");
//            Sequence();
//            matchToken(TokenType.RBRACKET, "ifStatement");
//            if (lookAhead.getType() == TokenType.KEY_ELSE) {
//                OptionalElse();
//            }
//        } else {
//            System.out.println("In OptionalElse();-----jmzhang-----3");
//            String errorTypes = TokenType.KEY_IF.toString();
//            parsingError(errorTypes, "ifStatement");
//        }
//
//    }
//
//    /**
//     * 方法名：OptionalElse
//     * 创建人：xrzhang
//     * 时间：2018年5月23日-下午7:27:53
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void OptionalElse() {
//        System.out.println("In OptionalElse();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.KEY_ELSE) {
//            System.out.println("In OptionalElse();-----jmzhang-----1");
//            matchToken(TokenType.KEY_ELSE, "OptionalElse");
//            matchToken(TokenType.LBRACKET, "OptionalElse");
//            Sequence();
//            matchToken(TokenType.RBRACKET, "OptionalElse");
//        } else if (lookAhead.getType() == TokenType.RBRACKET
//                || lookAhead.getType() == TokenType.KEY_IF
//                || lookAhead.getType() == TokenType.KEY_WHILE
//                || lookAhead.getType() == TokenType.IDENTIFIER) {
//            //match epslion
//            System.out.println("In OptionalElse();-----jmzhang-----2");
//        } else {
//            System.out.println("In OptionalElse();-----jmzhang-----3");
//            String errorTypes = TokenType.KEY_ELSE.toString()
//                    + "," + TokenType.RBRACKET.toString()
//                    + "," + TokenType.KEY_IF.toString()
//                    + "," + TokenType.KEY_WHILE.toString()
//                    + "," + TokenType.IDENTIFIER.toString();
//            parsingError(errorTypes, "OptionalElse");
//        }
//    }
//
//    /**
//     * assignmentStatement =IDENTIFIER ASSIGN expression SEMICOLON
//     * A->id = E;
//     * 方法名：assignmentStatement
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午8:56:26
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void assignmentStatement() {
//        System.out.println("In assignmentStatement();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.IDENTIFIER) {
//            matchToken(TokenType.IDENTIFIER, "assignmentStatement");
//            matchToken(TokenType.ASSIGN, "assignmentStatement");
//            expression();
//            matchToken(TokenType.SEMICOLON, "assignmentStatement");
//        } else {
//            String errorTypes = TokenType.IDENTIFIER.toString();
//            parsingError(errorTypes, "assignmentStatement");
//        }
//    }
//
//    /**
//     * expression = term expression_1
//     * E->TE'
//     * 方法名：expression
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午9:00:40
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void expression() {
//        System.out.println("In expression();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.IDENTIFIER
//                || lookAhead.getType() == TokenType.LPAREN
//                || lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            term();
//            expression_1();
//        } else {
//            String errorTypes = TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString()
//                    + "," + TokenType.LPAREN.toString();
//            parsingError(errorTypes, "expression");
//        }
//    }
//
//    /**
//     * expression_1=PLUS term expression_1 | MINUS term expression_1 | epslin
//     * E'->TE' | -TE' | ε
//     * 方法名：expression_1
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午9:06:26
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void expression_1() {
//        System.out.println("In expression_1();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.PLUS) {
//            System.out.println("In expression_1();-----jmzhang-----1");
//            matchToken(TokenType.PLUS, "expression_1");
//            term();
//            expression_1();
//        } else if (lookAhead.getType() == TokenType.MINUS) {
//            System.out.println("In expression_1();-----jmzhang-----2");
//            matchToken(TokenType.MINUS, "expression_1");
//            term();
//            expression_1();
//        } else if (lookAhead.getType() == TokenType.SEMICOLON
//                || lookAhead.getType() == TokenType.LESS
//                || lookAhead.getType() == TokenType.GREATER
//                || lookAhead.getType() == TokenType.LESS_EQUAL
//                || lookAhead.getType() == TokenType.GREATER_EQUAL
//                || lookAhead.getType() == TokenType.NOT_EQUAL
//                || lookAhead.getType() == TokenType.EQUAL
//                || lookAhead.getType() == TokenType.LOGICAL_AND
//                || lookAhead.getType() == TokenType.LOGICAL_OR
//                || lookAhead.getType() == TokenType.RPAREN) {
//            System.out.println("In expression_1();-----jmzhang-----3");
//            //match epslin
//            //follow(E')={')',','}
//        } else {
//            System.out.println("In expression_1();-----jmzhang-----4");
//            String errorTypes = TokenType.PLUS.toString()
//                    + "," + TokenType.MINUS.toString()
//                    + "," + TokenType.SEMICOLON.toString()
//                    + "," + TokenType.LESS.toString()
//                    + "," + TokenType.GREATER.toString()
//                    + "," + TokenType.LESS_EQUAL.toString()
//                    + "," + TokenType.GREATER_EQUAL.toString()
//                    + "," + TokenType.NOT_EQUAL.toString()
//                    + "," + TokenType.EQUAL.toString();
//            parsingError(errorTypes, "expression_1");
//        }
//    }
//
//    /**
//     * term=factor term_1
//     * T->FT'
//     * 方法名：term
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午9:16:51
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void term() {
//        System.out.println("In term();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.IDENTIFIER
//                || lookAhead.getType() == TokenType.LPAREN
//                || lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            factor();
//            term_1();
//        } else {
//            String errorTypes = TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString()
//                    + "," + TokenType.LPAREN.toString();
//            parsingError(errorTypes, "term");
//        }
//    }
//
//    /**
//     * term_1=MULT factor term_1 | DIV factor term_1 | MOD factor term_1 | epslin
//     * T'->*FT' | /FT' |%FT'|ε
//     * 方法名：term_1
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午9:20:00
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void term_1() {
//        System.out.println("In term_1();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.TIMES) {
//            matchToken(TokenType.TIMES, "term_1");
//            factor();
//            term_1();
//        } else if (lookAhead.getType() == TokenType.DIVIDE) {
//            matchToken(TokenType.DIVIDE, "term_1");
//            factor();
//            term_1();
//        } else if (lookAhead.getType() == TokenType.REMAINDER) {
//            matchToken(TokenType.REMAINDER, "term_1");
//            factor();
//            term_1();
//        } else if (lookAhead.getType() == TokenType.PLUS
//                || lookAhead.getType() == TokenType.MINUS
//                || lookAhead.getType() == TokenType.SEMICOLON
//                || lookAhead.getType() == TokenType.LESS
//                || lookAhead.getType() == TokenType.GREATER
//                || lookAhead.getType() == TokenType.LESS_EQUAL
//                || lookAhead.getType() == TokenType.GREATER_EQUAL
//                || lookAhead.getType() == TokenType.NOT_EQUAL
//                || lookAhead.getType() == TokenType.EQUAL
//                || lookAhead.getType() == TokenType.LOGICAL_AND
//                || lookAhead.getType() == TokenType.LOGICAL_OR
//                || lookAhead.getType() == TokenType.RPAREN) {
//            //match e[slion
//            //follow(T')={'+','-',')',',')
//        } else {
//            String errorTypes = TokenType.TIMES.toString()
//                    + "," + TokenType.DIVIDE.toString()
//                    + "," + TokenType.REMAINDER.toString()
//                    + "," + TokenType.PLUS.toString()
//                    + "," + TokenType.MINUS.toString()
//                    + "," + TokenType.SEMICOLON.toString()
//                    + "," + TokenType.LESS.toString()
//                    + "," + TokenType.GREATER.toString()
//                    + "," + TokenType.LESS_EQUAL.toString()
//                    + "," + TokenType.GREATER_EQUAL.toString()
//                    + "," + TokenType.NOT_EQUAL.toString()
//                    + "," + TokenType.EQUAL.toString()
//                    + "," + TokenType.LOGICAL_AND.toString()
//                    + "," + TokenType.LOGICAL_OR.toString();
//            System.out.println("lookAhead.getType()" + lookAhead.getType());
//            parsingError(errorTypes, "term_1");
//
//        }
//
//    }
//
//    /**
//     * factor = LPAREN expression RPAREN | IDENTIFER|INTEGER_LITERAL
//     * F->(E)|id| number
//     * 方法名：factor
//     * 创建人：xrzhang
//     * 时间：2018年5月16日-下午9:29:47
//     * 邮件：jmzhang_15_cauc@163.com void
//     *
//     * @throws
//     * @since 1.0.0
//     */
//    private void factor() {
//        System.out.println("In factor();-----jmzhang-----");
//        if (lookAhead.getType() == TokenType.LPAREN) {
//            matchToken(TokenType.LPAREN, "factor");
//            expression();
//            matchToken(TokenType.RPAREN, "factor");
//        } else if (lookAhead.getType() == TokenType.IDENTIFIER) {
//            matchToken(TokenType.IDENTIFIER, "factor");
//        } else if (lookAhead.getType() == TokenType.INTEGER_LITERAL) {
//            matchToken(TokenType.INTEGER_LITERAL, "factor");
//        } else {
//            String errorTypes = TokenType.LPAREN.toString()
//                    + "," + TokenType.IDENTIFIER.toString()
//                    + "," + TokenType.INTEGER_LITERAL.toString();
//            parsingError(errorTypes, "factor");
//        }
//    }
//}
