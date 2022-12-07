package com.deep.framework.ast.node;

import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.lexer.TokenType;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.*;

public interface AssignNode {
    List<TokenType> ASSIGN_TYPE = Stream.of(ASSIGN, ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, AND_ASSIGN, OR_ASSIGN, XOR_ASSIGN, MOD_ASSIGN, LSHIFT_ASSIGN, RSHIFT_ASSIGN, URSHIFT_ASSIGN);

    Expression getVariable();

    Expression getValue();

    TokenType getOperator();
}