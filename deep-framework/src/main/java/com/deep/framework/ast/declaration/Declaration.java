package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.*;

public class Declaration extends Expression {

    private Name name;

    protected static List<TokenType> Method_Modifiers = List.of(PUBLIC, PROTECTED, PRIVATE, STATIC, FINAL, ABSTRACT, DEFAULT, SYNCHRONIZED);
    protected static List<TokenType> Field_Modifiers = List.of(PUBLIC, PROTECTED, PRIVATE, STATIC, FINAL, VOLATILE, TRANSIENT);

    public Declaration(Node prarent) {super(prarent);}

    public void setName(Name name) {
        this.name = name;
    }
}
