package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.lexer.TokenType;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.*;

public class Declaration extends Expression {
    public static List<TokenType> Method_Modifiers = Stream.of(PUBLIC, PROTECTED, PRIVATE, STATIC, FINAL, ABSTRACT, DEFAULT, SYNCHRONIZED);
    public static List<TokenType> Field_Modifiers = Stream.of(PUBLIC, PROTECTED, PRIVATE, STATIC, FINAL, VOLATILE, TRANSIENT);

    public Declaration(Node prarent) {
        super(prarent);
    }

    public static List<TokenType> is() {
        return Field_Modifiers;
    }
}